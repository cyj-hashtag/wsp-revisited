"""RQ2: can an LLM-free procedure find an edit like p50c? (docs/plan-for-agents.md section 3, RQ2.) The rules that
choose the edit are fixed in prereg/rq2_rules.md before the run; no benchmark takes part in any choice, and no LLM is
called anywhere.

    python -m wsprev.rq2 <step> [options]          (scripts/run_stage.py runs the GPU steps as stages, on every GPU)

  plan lean|full   fix the plan (results/rq2/plan.json; commit it before stage rq2-dev)
  [rq2-dev]        stage: base, D0, practice DEV[:dev_problems], 1 sample (wsprev.generate)               GPU
  read             teacher-forced activations of the windows (HF transformers)              stage rq2-read, GPU
  build            directions, AUC, neuron scores, the grid of edits -> results/rq2/build.json            CPU*
  swapcheck        an edit swapped into a running engine equals the same weights loaded from disk         GPU
  screen           every edit of the grid (and base, p50c for reference) on DEV[448:512], D0, at most 8,192
                   tokens                                                                 stage rq2-screen, GPU
  choose screen    rank the edits -> results/rq2/screen.json (the top ones go to HOLDOUT)                  CPU
  holdout          the top edits on HOLDOUT[:holdout_problems], D0, at most 32,768 tokens stage rq2-holdout, GPU
  choose holdout   the final edit -> results/rq2/choice.json                                               CPU
  export           model folders rq2-auto and rq2-ctrl-a/b/c under $WSPREV_MODELS                         CPU*
  [rq2-final]      stage: MATH-500 runs of rq2-auto (D0; D3 in the full plan) and of the controls (D0)   GPU
  report           results/rq2/rq2.json and results/rq2_summary.md (needs RQ1's runs/d0 and runs/rq1)      CPU
  (* reads the base model's weights)

Raw output under $WSPREV_RUNS/rq2/ (not in git); small results under results/rq2/.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import data, decoding
from . import directions as dr
from ._vendor import REPO
from .summary import group, metrics, read_records

RESULTS = REPO / "results" / "rq2"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"     # the weight file of the model folders (setup.sh)
SWAP_EXT = "wsprev._vllm_swap.WeightSwap"
SCREEN_PROBLEMS = (448, 512)        # practice DEV indices of the screening problems (never used for directions)
SCREEN_MAX_TOKENS = 8192
REFERENCES = ("base", "p50c")       # run in the screening for comparison; never chosen
CONTROL_SEED = 20261105
D_FF = 3584
PLANS = {
    "full": {"dev_problems": 448, "max_pairs": 256, "ks": dr.KS, "alphas": dr.ALPHAS, "top": 5,
             "holdout_problems": 256, "final": (("D0", 1), ("D3", 2)), "controls": ("a", "b", "c"),
             "control_problems": 500},
    "lean": {"dev_problems": 160, "max_pairs": 96, "ks": (64, 3584), "alphas": (1, 4, 6), "top": 2,
             "holdout_problems": 128, "final": (("D0", 1),), "controls": ("a", "b", "c"), "control_problems": 100},
}


def runs_dir() -> Path:
    return Path(os.environ.get("WSPREV_RUNS", data.home() / "runs")) / "rq2"


def models_dir() -> Path:
    return Path(os.environ.get("WSPREV_MODELS", data.home() / "models"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {path}")


def plan(results: Path | None = None) -> dict:
    p = (results or RESULTS) / "plan.json"
    if not p.exists():
        raise SystemExit("no RQ2 plan: python -m wsprev.rq2 plan lean|full (then commit results/rq2/plan.json)")
    name = load_json(p)["plan"]
    return {"name": name, **PLANS[name]}


def shard_of(spec: str | None) -> tuple[int, int]:
    if not spec:
        return 0, 1
    i, k = map(int, spec.split("/"))
    return i, k


# ---------------------------------------------------------------- weights
def load_down_proj(model_dir, layers) -> dict[int, np.ndarray]:
    """{layer: down_proj weight as float32} from a model folder's safetensors file."""
    sys.path.insert(0, str(REPO / "third_party" / "wsp_logs"))
    import st_numpy as S  # noqa: PLC0415

    names = {dr.TENSOR.format(l): l for l in layers}
    t = S.load(str(Path(model_dir) / WEIGHTS), set(names))
    if set(t) != set(names):
        raise RuntimeError(f"missing tensors in {model_dir}: {sorted(set(names) - set(t))}")
    return {names[n]: S.bf16_to_f32(bits).astype(np.float32) for n, (dt, bits) in t.items() if dt == "BF16"}


def bf16_bits(x32) -> np.ndarray:
    sys.path.insert(0, str(REPO / "third_party" / "wsp_logs"))
    import st_numpy as S  # noqa: PLC0415

    return S.bf16_bits(np.asarray(x32, dtype=np.float32))


def to_torch_bf16(bits: np.ndarray):
    import torch  # noqa: PLC0415

    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16).copy()).view(torch.bfloat16)


def same_outside(path_a, path_b, ranges, chunk=1 << 24) -> bool:
    """True if two files have the same size and the same bytes outside `ranges` ([start, end) pairs)."""
    size = os.path.getsize(path_a)
    if size != os.path.getsize(path_b):
        return False
    cuts = [0]
    for start, end in sorted(ranges):
        cuts += [start, end]
    cuts.append(size)
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        for start, end in zip(cuts[0::2], cuts[1::2]):
            fa.seek(start)
            fb.seek(start)
            pos = start
            while pos < end:
                n = min(chunk, end - pos)
                if fa.read(n) != fb.read(n):
                    return False
                pos += n
    return True


def sha256_file(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_model(base_dir, tensors: dict[str, np.ndarray], out_dir, info: dict) -> dict:
    """A model folder equal to base_dir except for the given bfloat16 tensors {name: uint16 bits}, as
    scripts/build_p50c.py builds p50c: the base weight file is copied and only those tensors' bytes are overwritten;
    every other byte is checked to equal the base file. Writes wsprev_model.json (name, weight_sha256, info)."""
    sys.path.insert(0, str(REPO / "third_party" / "wsp_logs"))
    import st_numpy as S  # noqa: PLC0415

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    head, off = S.read_header(base_dir / WEIGHTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():
        if f.is_file() and f.name not in (WEIGHTS, "wsprev_model.json"):
            shutil.copy2(f, out_dir / f.name)
    tmp = out_dir / (WEIGHTS + ".tmp")
    shutil.copyfile(base_dir / WEIGHTS, tmp)
    ranges, changed = [], {}
    with open(tmp, "r+b") as fo:
        for name, bits in sorted(tensors.items()):
            h = head[name]
            if h["dtype"] != "BF16" or list(bits.shape) != h["shape"]:
                raise ValueError(f"{name}: {h['dtype']} {h['shape']} in the file, uint16 bits {bits.shape} given")
            start, end = off + h["data_offsets"][0], off + h["data_offsets"][1]
            fo.seek(start)
            old = np.frombuffer(fo.read(end - start), dtype=np.uint16)
            changed[name] = int((old != bits.reshape(-1)).sum())
            fo.seek(start)
            fo.write(np.ascontiguousarray(bits, dtype=np.uint16).tobytes())
            ranges.append((start, end))
    if not same_outside(base_dir / WEIGHTS, tmp, ranges):
        raise RuntimeError("bytes outside the edited tensors differ from the base file")
    os.replace(tmp, out_dir / WEIGHTS)
    meta = {"name": info["name"], "weight_sha256": sha256_file(out_dir / WEIGHTS), "changed_entries": changed,
            **{k: v for k, v in info.items() if k != "name"}}
    (out_dir / "wsprev_model.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"{info['name']}: {sum(changed.values()):,} entries changed in {len(changed)} tensors -> {out_dir}")
    return meta


# ---------------------------------------------------------------- edits from results/rq2/build.json
def edit_by_name(build: dict, name: str) -> dict:
    for e in build["edits"]:
        if e["name"] == name:
            return e
    raise KeyError(name)


def edit_mask(build: dict, e: dict, layer: int) -> np.ndarray:
    """0/1 mask of the edit's columns at a layer: its top-k columns by c_j (build.json), or every column."""
    if e["k"] >= D_FF:
        return np.ones(D_FF)
    m = np.zeros(D_FF)
    m[build["top_columns"][str(layer)][str(e["k"])]] = 1.0
    return m


def edit_weights(build: dict, e: dict, W: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """{layer: W' as float32} of an edit of the grid (its u and top-k columns are in build.json)."""
    u = {l: np.array(build["u"][str(l)]) for l in e["layers"]}
    m = {l: edit_mask(build, e, l) for l in e["layers"]}
    return dr.edited({l: W[l] for l in e["layers"]}, u, m, e["alpha"])


# ---------------------------------------------------------------- step: read (GPU, HF transformers)
def dev_records(dev_runs: Path, ids: list[str]) -> tuple[list[dict], dict]:
    recs, _ = read_records([dev_runs])
    texts = {}
    for p in sorted(dev_runs.glob("**/texts.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    texts[(t["problem_id"], t["sample_idx"])] = t["text"]
    pos = {i: n for n, i in enumerate(ids)}
    recs = sorted((r for r in recs if r["problem_id"] in pos), key=lambda r: (pos[r["problem_id"]], r["sample_idx"]))
    return recs, texts


def cmd_read(a):
    p = plan()
    i, k = shard_of(a.shard)
    rows = data.load("practice:dev")[:p["dev_problems"]]
    by_id = {r["id"]: r for r in rows}
    recs, texts = dev_records(runs_dir() / "base_D0_practice-dev", [r["id"] for r in rows])
    if len(recs) != len(rows):
        raise SystemExit(f"{len(recs)} DEV records for {len(rows)} problems: run stage rq2-dev first")
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from .generate import resolve_model  # noqa: PLC0415

    _, base_dir, _ = resolve_model("base")
    tok = AutoTokenizer.from_pretrained(base_dir)

    def prompt_ids(pid):
        text = tok.apply_chat_template([{"role": "user", "content": data.user_message(by_id[pid])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        return tok.encode(text)

    ws, why = dr.windows(recs, texts, lambda t: tok.encode(t, add_special_tokens=False), prompt_ids,
                         max_pairs=p["max_pairs"])
    mine = [w for n, w in enumerate(ws) if (w.pair if w.pair is not None else n) % k == i]
    print(f"{len(ws)} windows ({why}); shard {i}/{k}: {len(mine)}", flush=True)
    if a.max_context:     # fallback if a full context does not fit: keep the window and the tokens just before it
        for w in mine:
            drop = max(0, len(w.ids) - a.max_context)
            if drop:
                w.ids, w.start = w.ids[drop:], w.start - drop
    model = AutoModel.from_pretrained(base_dir, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    reader = dr.Reader(model)
    res, act, t0 = [], [], time.time()
    for n, w in enumerate(mine):
        r_, a_ = reader.read(w.ids, w.start)
        res.append(r_)
        act.append(a_)
        if n % 20 == 0:
            print(f"[{(time.time() - t0) / 60:5.1f} min] {n + 1}/{len(mine)} windows, context {len(w.ids):,} tokens, "
                  f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    out = runs_dir() / "read"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"s{i}of{k}.npz", res=np.stack(res), act=np.stack(act), kind=np.array([w.kind for w in mine]),
             pair=np.array([-1 if w.pair is None else w.pair for w in mine]),
             problem_id=np.array([w.problem_id for w in mine]), dup8=np.array([w.dup8 for w in mine]),
             context=np.array([len(w.ids) for w in mine]), layers=np.array(dr.LAYERS))
    write_json(out / f"s{i}of{k}.json", {"windows": len(mine), "skipped": why, "seconds": time.time() - t0,
                                          "max_context": a.max_context,
                                          "peak_gib": torch.cuda.max_memory_allocated() / 2**30})


# ---------------------------------------------------------------- step: build (CPU)
def load_read(read_dir: Path) -> dict:
    parts = [np.load(p) for p in sorted(read_dir.glob("s*of*.npz"))]
    if not parts:
        raise SystemExit(f"no activations under {read_dir}: run stage rq2-read first")
    return {key: np.concatenate([q[key] for q in parts]) for key in ("res", "act", "kind", "pair", "problem_id",
                                                                     "dup8", "context")} | {"layers": parts[0]["layers"]}


def build(read: dict, W: dict[int, np.ndarray], p: dict) -> dict:
    """Directions, separability, scores and the grid (module docstring of wsprev.directions)."""
    layers = tuple(int(x) for x in read["layers"])
    loop, norm = read["kind"] == "loop", read["kind"] == "normal"
    pl, pn = read["pair"][loop], read["pair"][norm]
    if sorted(pl) != sorted(pn):
        raise ValueError("loop and normal windows do not pair up")
    ol, on = np.argsort(pl), np.argsort(pn)
    res_l, res_n = read["res"][loop][ol], read["res"][norm][on]
    act_l, act_n = read["act"][loop][ol].mean(0), read["act"][norm][on].mean(0)
    d = dr.directions(res_l, res_n)
    auc_layers = dr.top_layers(layers, d["auc"], d["separation"])
    fin = read["kind"] == "finished"
    edits = dr.grid(auc_layers, p["ks"], p["alphas"])
    used = sorted({l for e in edits for l in e.layers})
    out = {"plan": p["name"], "pairs": int(len(pl)), "finished_windows": int(fin.sum()),
           "layers": {}, "auc_layers": list(auc_layers), "astra_layers": list(dr.ASTRA_LAYERS),
           "edits": [e.to_dict() for e in edits], "u": {}, "top_columns": {}, "prop3": {}}
    c_all = {}
    for n, l in enumerate(layers):
        u = d["u"][n]
        row = {"auc": float(d["auc"][n]), "separation": float(d["separation"][n]),
               "mean_gap_norm": float(np.linalg.norm(res_l[:, n].mean(0) - res_n[:, n].mean(0)))}
        if fin.any():
            row["auc_loop_vs_finished"] = dr.auc(res_l[:, n] @ u, read["res"][fin][:, n] @ u)
        if l in W:
            c = dr.scores(W[l], u, act_l[n], act_n[n])
            c_all[l] = c
            row.update(c_positive=int((c > 0).sum()), c_sum_positive=float(c[c > 0].sum()))
        out["layers"][str(l)] = row
    for l in used:
        n = layers.index(l)
        out["u"][str(l)] = [float(x) for x in d["u"][n]]
        out["top_columns"][str(l)] = {str(k): [int(j) for j in np.argsort(-c_all[l], kind="stable")[:k]]
                                      for k in p["ks"] if k < D_FF}
    for e in edits:
        n_ = {l: layers.index(l) for l in e.layers}
        m = dr.masks({l: c_all[l] for l in e.layers}, e.k)
        out["prop3"][e.name] = {str(l): dr.projection_gap_change(W[l], d["u"][n_[l]], m[l], e.alpha, act_l[n_[l]],
                                                                 act_n[n_[l]]) for l in e.layers}
    return out


def cmd_build(a):
    p = plan()
    read = load_read(runs_dir() / "read")
    from .generate import resolve_model  # noqa: PLC0415

    _, base_dir, _ = resolve_model("base")
    W = load_down_proj(base_dir, tuple(int(x) for x in read["layers"]))
    write_json(RESULTS / "build.json", build(read, W, p))


# ---------------------------------------------------------------- the swap runner (GPU, vLLM)
def run_edits(engine, tok, edits, rows, cfg, out_root: Path, tag: str | None, rpc, log=print):
    """Generate `rows` (1 sample) with every edit in turn on one engine. edits: [(name, {layer: bf16 tensor} or None,
    info)]; rpc(method, *args) calls the worker extension (wsprev._vllm_swap.WeightSwap). Output folders
    out_root/<name>[/<tag>] in wsprev.generate's format; finished samples are skipped on a re-run."""
    from . import grade  # noqa: PLC0415
    from .generate import Output, Runner, record_of  # noqa: PLC0415

    n_close = len(tok.encode(decoding.CLOSE_THINK, add_special_tokens=False))
    for name, weights, info in edits:
        folder = out_root / name / tag if tag else out_root / name
        job = {"model": name, "edit": info, "config": cfg.to_dict(), "benchmark": info.get("benchmark"),
               "samples": 1, "shard": tag}
        out = Output(folder, job)
        jobs = [(r, 0) for r in rows if (r["id"], 0) not in out.done]
        log(f"{name}: {len(jobs)} samples to generate, {len(out.done)} done", flush=True)
        if not jobs:
            continue
        rpc("wsprev_restore")
        if weights:
            rpc("wsprev_set_down_proj", weights)
        engine.llm.reset_prefix_cache()
        base = {"benchmark": info.get("benchmark"), "model": name, "model_sha": None, "config": cfg.name}

        def on_done(s, reason, out=out, base=base):
            rec, text = record_of(s, cfg, n_close, reason, base)
            try:
                rec.update(grade.grade(s.row, text, cfg.thinking))
            except Exception as e:  # noqa: BLE001
                rec.update({"pred": None, "correct": False, "grade_error": repr(e)[:300]})
            out.write(rec, text)

        session = out.start_session({"argv": sys.argv, "edit": name})
        stats = Runner(engine, tok, cfg, on_done, log=log).run(jobs)
        out.end_session(session, stats)
    rpc("wsprev_restore")


def swap_engine():
    from .generate import VLLMEngine, resolve_model  # noqa: PLC0415

    _, base_dir, _ = resolve_model("base")
    engine = VLLMEngine(base_dir, extra_args={"worker_extension_cls": SWAP_EXT})

    def rpc(method, *args):
        return engine.llm.collective_rpc(method, args=args)

    from transformers import AutoTokenizer  # noqa: PLC0415

    return engine, AutoTokenizer.from_pretrained(base_dir), rpc, base_dir


def p50c_weights() -> dict[int, np.ndarray]:
    from .generate import resolve_model  # noqa: PLC0415

    _, d, _ = resolve_model("p50c")
    return load_down_proj(d, dr.ASTRA_LAYERS)


def screen_edits(build_: dict, W: dict, names=None) -> list:
    """[(name, {layer: bf16 tensor} or None, info)]: the references, then the grid (or the named edits)."""
    out = []
    todo = list(REFERENCES) + [e["name"] for e in build_["edits"]] if names is None else list(names)
    for name in todo:
        if name == "base":
            out.append((name, None, {"reference": "base"}))
        elif name == "p50c":
            out.append((name, {l: to_torch_bf16(bf16_bits(w)) for l, w in p50c_weights().items()},
                        {"reference": "p50c"}))
        else:
            e = edit_by_name(build_, name)
            out.append((name, {l: to_torch_bf16(bf16_bits(w)) for l, w in edit_weights(build_, e, W).items()}, e))
    return out


def cmd_screen(a):
    plan()
    i, k = shard_of(a.shard)
    check = RESULTS / "swapcheck.json"
    if not check.exists() or not load_json(check).get("pass"):
        raise SystemExit("the swap check has not passed (python -m wsprev.rq2 swapcheck)")
    b = load_json(RESULTS / "build.json")
    rows = data.load("practice:dev")[SCREEN_PROBLEMS[0]:SCREEN_PROBLEMS[1]]
    engine, tok, rpc, base_dir = swap_engine()
    W = load_down_proj(base_dir, sorted({l for e in b["edits"] for l in e["layers"]}))
    names = (list(REFERENCES) + [e["name"] for e in b["edits"]])[i::k]
    edits = [(n, w, dict(info, benchmark="practice:dev-screen")) for n, w, info in screen_edits(b, W, names)]
    cfg = replace(decoding.D0, name="D0-8k", max_tokens=SCREEN_MAX_TOKENS)
    run_edits(engine, tok, edits, rows, cfg, runs_dir() / "screen", None, rpc)


def cmd_holdout(a):
    p = plan()
    i, k = shard_of(a.shard)
    b, s = load_json(RESULTS / "build.json"), load_json(RESULTS / "screen.json")
    rows = data.load("practice:holdout")[:p["holdout_problems"]][i::k]
    engine, tok, rpc, base_dir = swap_engine()
    W = load_down_proj(base_dir, sorted({l for e in b["edits"] for l in e["layers"]}))
    edits = [(n, w, dict(info, benchmark="practice:holdout")) for n, w, info in screen_edits(b, W, s["top"])]
    run_edits(engine, tok, edits, rows, decoding.D0, runs_dir() / "holdout", f"s{i}of{k}", rpc)


# ---------------------------------------------------------------- step: swapcheck (GPU, vLLM)
SWAPCHECK_PROBLEMS, SWAPCHECK_TOKENS = 4, 48


def greedy(engine, prompts) -> list[list[int]]:
    from vllm import SamplingParams  # noqa: PLC0415

    outs = engine.llm.generate([{"prompt_token_ids": p} for p in prompts],
                               SamplingParams(temperature=0.0, max_tokens=SWAPCHECK_TOKENS), use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in outs]


def check_prompts(tok) -> list[list[int]]:
    rows = data.load("practice:select")[:SWAPCHECK_PROBLEMS]
    return [tok.encode(tok.apply_chat_template([{"role": "user", "content": data.user_message(r)}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)) for r in rows]


def cmd_swapcheck(a):
    from transformers import AutoTokenizer  # noqa: PLC0415

    from .generate import VLLMEngine, resolve_model  # noqa: PLC0415

    ref_path = runs_dir() / "swapcheck_p50c.json"
    if a.reference:   # p50c loaded from disk
        _, d, _ = resolve_model("p50c")
        engine = VLLMEngine(d)
        write_json(ref_path, {"tokens": greedy(engine, check_prompts(AutoTokenizer.from_pretrained(d)))})
        return
    engine, tok, rpc, base_dir = swap_engine()
    prompts = check_prompts(tok)
    g0 = greedy(engine, prompts)
    rpc("wsprev_set_down_proj", {l: to_torch_bf16(bf16_bits(w)) for l, w in p50c_weights().items()})
    engine.llm.reset_prefix_cache()
    g1 = greedy(engine, prompts)
    rpc("wsprev_restore")
    engine.llm.reset_prefix_cache()
    g2 = greedy(engine, prompts)
    import torch  # noqa: PLC0415

    base_w = load_down_proj(base_dir, dr.ASTRA_LAYERS)
    restored_ok = all(bool(torch.equal(rpc("wsprev_get_down_proj", l)[0].float(), torch.from_numpy(base_w[l])))
                      for l in dr.ASTRA_LAYERS)
    ref = load_json(ref_path)["tokens"] if ref_path.exists() else None
    res = {"swapped_equals_p50c_from_disk": g1 == ref if ref is not None else None,
           "restored_equals_base": g2 == g0, "swap_changes_output": g1 != g0,
           "restored_weight_bit_identical": restored_ok, "prompts": len(prompts), "tokens": SWAPCHECK_TOKENS}
    res["pass"] = bool(res["swapped_equals_p50c_from_disk"] and res["restored_equals_base"]
                       and res["swap_changes_output"] and restored_ok)
    write_json(RESULTS / "swapcheck.json", res)
    print("SWAP CHECK", "PASS" if res["pass"] else "FAIL", res)
    if not res["pass"]:
        sys.exit(1)


# ---------------------------------------------------------------- step: choose (CPU)
def rank_key(e: dict, row: dict) -> tuple:
    """Order of prereg/rq2_rules.md: more correct, then more finished, then smaller alpha, smaller k, Astra's layers
    before the AUC layers."""
    return (-row["correct"], -row["finished"], e["alpha"], e["k"], 0 if e["layer_set"] == "astra" else 1)


def tally(folder: Path, expected: int) -> dict | None:
    if not folder.exists() or not list(folder.glob("**/records.jsonl")):
        return None
    recs, metas = read_records([folder])
    if len(recs) != expected:
        raise SystemExit(f"{folder}: {len(recs)} samples, expected {expected}")
    m = metrics(recs)
    secs = sum(s.get("seconds") or 0 for mt in metas for s in mt["sessions"])
    toks = sum(s.get("tokens") or 0 for mt in metas for s in mt["sessions"])
    return {"samples": len(recs), "correct": int(sum(r["correct"] for r in recs)),
            "finished": int(sum(not r["cut"] for r in recs)), "accuracy": m["accuracy"], "cut_share": m["cut_share"],
            "mean_tokens": m["mean_tokens"], "gpu_seconds": secs, "tokens": toks}


def choose(step: str, build_: dict, p: dict, root: Path | None = None) -> dict:
    root = root or runs_dir()
    if step == "screen":
        n, names = SCREEN_PROBLEMS[1] - SCREEN_PROBLEMS[0], [e["name"] for e in build_["edits"]]
        table = {name: tally(root / "screen" / name, n) for name in list(REFERENCES) + names}
        missing = [k for k, v in table.items() if v is None]
        if missing:
            raise SystemExit(f"screening incomplete: {missing}")
        ranked = sorted(names, key=lambda x: rank_key(edit_by_name(build_, x), table[x]))
        return {"plan": p["name"], "problems": n, "max_tokens": SCREEN_MAX_TOKENS, "table": table,
                "ranking": ranked, "top": ranked[:p["top"]]}
    s = load_json(RESULTS / "screen.json")
    table = {name: tally(root / "holdout" / name, p["holdout_problems"]) for name in s["top"]}
    missing = [k for k, v in table.items() if v is None]
    if missing:
        raise SystemExit(f"holdout incomplete: {missing}")
    ranked = sorted(s["top"], key=lambda x: rank_key(edit_by_name(build_, x), table[x]))
    return {"plan": p["name"], "problems": p["holdout_problems"], "table": table, "ranking": ranked,
            "choice": ranked[0], "edit": edit_by_name(build_, ranked[0])}


def cmd_choose(a):
    p = plan()
    res = choose(a.step, load_json(RESULTS / "build.json"), p)
    write_json(RESULTS / ("screen.json" if a.step == "screen" else "choice.json"), res)


# ---------------------------------------------------------------- step: export (CPU)
def controls(build_: dict, e: dict, W: dict, W_p50c: dict | None, which=("a", "b", "c")) -> dict:
    """{control name: ({layer: W' float32}, info)} (plan section 3, RQ2 step 8):
    a: the chosen edit's columns and alpha with a uniformly random unit direction per layer;
    b: its direction and alpha with k random columns per layer (skipped if k = d_ff: identical to the edit);
    c: p50c's columns and direction (recovered from its weights) with alpha = 1, on p50c's layers."""
    rng = np.random.default_rng(CONTROL_SEED)
    out = {}
    mask = {l: edit_mask(build_, e, l) for l in e["layers"]}
    u = {l: np.array(build_["u"][str(l)]) for l in e["layers"]}
    if "a" in which:
        from .edits import random_unit  # noqa: PLC0415

        ua = {l: random_unit(len(u[l]), rng) for l in e["layers"]}
        out["rq2-ctrl-a"] = (dr.edited(W, ua, mask, e["alpha"]), {"control": "a", "of": e["name"],
                                                                 "seed": CONTROL_SEED})
    if "b" in which and e["k"] < D_FF:
        from .edits import random_mask  # noqa: PLC0415

        mb = {l: random_mask(D_FF, e["k"], rng) for l in e["layers"]}
        out["rq2-ctrl-b"] = (dr.edited(W, u, mb, e["alpha"]), {"control": "b", "of": e["name"], "seed": CONTROL_SEED})
    if "c" in which and W_p50c is not None:
        uc, mc, check = {}, {}, {}
        for l in dr.ASTRA_LAYERS:
            uc[l], mc[l] = dr.recover_rank1(W[l], W_p50c[l], 6.0)
            rebuilt = bf16_bits(dr.edited({l: W[l]}, {l: uc[l]}, {l: mc[l]}, 6.0)[l])
            check[str(l)] = {"columns": int(mc[l].sum()),
                             "alpha6_rebuild_equal_share": float((rebuilt == bf16_bits(W_p50c[l])).mean())}
        out["rq2-ctrl-c"] = (dr.edited(W, uc, mc, 1.0), {"control": "c", "of": "p50c", "alpha": 1.0,
                                                          "recovery_check": check})
    return out


def cmd_export(a):
    p = plan()
    from .generate import resolve_model  # noqa: PLC0415

    b, ch = load_json(RESULTS / "build.json"), load_json(RESULTS / "choice.json")
    e = ch["edit"]
    _, base_dir, _ = resolve_model("base")
    layers = sorted(set(e["layers"]) | set(dr.ASTRA_LAYERS))
    W = load_down_proj(base_dir, layers)
    _, p50c_dir, _ = resolve_model("p50c")
    made = {"rq2-auto": write_model(base_dir, {dr.TENSOR.format(l): bf16_bits(w)
                                               for l, w in edit_weights(b, e, W).items()},
                                    models_dir() / "rq2-auto", {"name": "rq2-auto", "edit": e})}
    for name, (ws, info) in controls(b, e, W, load_down_proj(p50c_dir, dr.ASTRA_LAYERS), p["controls"]).items():
        made[name] = write_model(base_dir, {dr.TENSOR.format(l): bf16_bits(w) for l, w in ws.items()},
                                 models_dir() / name, {"name": name, **info})
    write_json(RESULTS / "models.json", made)


# ---------------------------------------------------------------- step: report (CPU)
def paired_delta(groups, a_key, b_key, n_boot=10_000) -> dict | None:
    """Acc(a) - Acc(b) on their common problems with the 95% percentile interval of the paired bootstrap over
    problems (sorted ids, default_rng(0), as in prereg section 5)."""
    if a_key not in groups or b_key not in groups:
        return None
    per = {}
    for key in (a_key, b_key):
        d = defaultdict(list)
        for r in groups[key]:
            d[r["problem_id"]].append(r["correct"])
        per[key] = d
    ids = sorted(set(per[a_key]) & set(per[b_key]))
    x = np.array([np.mean(per[a_key][i]) for i in ids]) - np.array([np.mean(per[b_key][i]) for i in ids])
    idx = np.random.default_rng(0).integers(0, len(ids), size=(n_boot, len(ids)))
    boot = x[idx].mean(axis=1)
    return {"problems": len(ids), "estimate": float(x.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def cmd_report(a):
    p = plan()
    runs = Path(os.environ.get("WSPREV_RUNS", data.home() / "runs"))
    recs = []
    for sub in ("d0", "rq1"):
        if (runs / sub).exists():
            recs += read_records([runs / sub])[0]
    final = runs_dir() / "final"
    if final.exists():
        recs += read_records([final])[0]
    g = {k: v for k, v in group(recs).items() if k[2] == "math500"}
    rows = {f"{m}/{c}": metrics(rs) for (m, c, _), rs in sorted(g.items())}
    comps = {}
    for c, _ in p["final"]:
        for other in ("base", "p50c"):
            comps[f"rq2-auto - {other} ({c})"] = paired_delta(g, ("rq2-auto", c, "math500"), (other, c, "math500"))
    for ctl in ("rq2-ctrl-a", "rq2-ctrl-b", "rq2-ctrl-c"):
        comps[f"{ctl} - base (D0)"] = paired_delta(g, (ctl, "D0", "math500"), ("base", "D0", "math500"))
    cost = {"llm_tokens": 0, "gpu_seconds": 0.0, "generated_tokens": 0}
    for meta in runs_dir().glob("**/meta.json"):
        for s in load_json(meta)["sessions"]:
            cost["gpu_seconds"] += s.get("seconds") or 0
            cost["generated_tokens"] += s.get("tokens") or 0
    for js in (runs_dir() / "read").glob("s*of*.json"):
        cost["gpu_seconds"] += load_json(js)["seconds"]
    out = {"plan": p, "math500": rows, "comparisons": comps, "cost": cost}
    for name in ("build", "screen", "choice", "swapcheck", "models"):
        f = RESULTS / f"{name}.json"
        if f.exists():
            out[name] = load_json(f)
    write_json(RESULTS / "rq2.json", out)
    L = ["# RQ2 results: an LLM-free edit", "", f"Plan: {p['name']} (prereg/rq2_rules.md). "
         f"Computed by `python -m wsprev.rq2 report` on {time.strftime('%Y-%m-%d %H:%M')}. No LLM was called.", ""]
    if "build" in out:
        b = out["build"]
        L += ["## Directions", "", f"{b['pairs']} loop/normal pairs. Layers by cross-fitted AUC: "
              f"{b['auc_layers']} (Astra's: {b['astra_layers']}).", "", "| Layer | AUC | separation |", "|---|---|---|"]
        L += [f"| {l} | {v['auc']:.3f} | {v['separation']:.2f} |" for l, v in b["layers"].items()]
        L.append("")
    if "screen" in out:
        s = out["screen"]
        L += [f"## Screening ({s['problems']} DEV problems, D0, at most {s['max_tokens']:,} tokens)", "",
              "| Edit | Correct | Finished | Mean tokens |", "|---|---|---|---|"]
        L += [f"| {n} | {s['table'][n]['correct']} | {s['table'][n]['finished']} | {s['table'][n]['mean_tokens']:,.0f} |"
              for n in list(REFERENCES) + s["ranking"]]
        L += ["", f"Top: {', '.join(s['top'])}", ""]
    if "choice" in out:
        c = out["choice"]
        L += [f"## HOLDOUT ({c['problems']} problems, D0)", "", "| Edit | Correct | Finished | Mean tokens |",
              "|---|---|---|---|"]
        L += [f"| {n} | {c['table'][n]['correct']} | {c['table'][n]['finished']} | {c['table'][n]['mean_tokens']:,.0f} |"
              for n in c["ranking"]]
        L += ["", f"Chosen: **{c['choice']}**", ""]
    L += ["## MATH-500", "", "| Model / config | Samples | Accuracy | Cut | Mean tokens |", "|---|---|---|---|---|"]
    L += [f"| {k} | {m['samples']} | {pct(m['accuracy'])} | {pct(m['cut_share'])} | {m['mean_tokens']:,.0f} |"
          for k, m in rows.items()]
    L += ["", "| Comparison | Problems | Difference (pp) | 95% CI (pp) |", "|---|---|---|---|"]
    for k, v in comps.items():
        if v:
            L.append(f"| {k} | {v['problems']} | {100 * v['estimate']:+.1f} | "
                     f"[{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}] |")
    L += ["", f"Cost: {cost['gpu_seconds'] / 3600:.2f} GPU hours (sum over GPUs), "
          f"{cost['generated_tokens']:,} generated tokens, 0 LLM tokens.", ""]
    (REPO / "results" / "rq2_summary.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("\n".join(L))


# ---------------------------------------------------------------- cost estimate (CPU, before renting)
YUAN_PER_HOUR = 3.0       # the 2 x RTX 3090 machine of 2026-11-04 (plan section 4)
SETUP_HOURS = 1.0         # rough: setup.sh again if the data disk was wiped
ENGINE_LOAD_MIN = 2.0     # per vLLM engine start (rough)


def estimate(name: str, results: Path | None = None, gpus: int = 2) -> dict:
    """Generated tokens and hours of a plan from measured RQ1 numbers: per-GPU throughput of the D0 calibration
    (results/calib/*/summary.json) and mean tokens per sample on MATH-500 (results/rq1/rq1.json). An edit's samples are
    assumed to be as long as p50c's (low) to base's (high)."""
    results = results or (REPO / "results")
    tput = [load_json(results / "calib" / f"{m}_D0_math500" / "summary.json")["throughput"]["tokens_per_s"]
            for m in ("base", "p50c")]
    runs = load_json(results / "rq1" / "rq1.json")["runs"]
    mean = {k: runs[k]["mean_tokens"] for k in ("base/D3/math500", "p50c/D3/math500")}
    d0 = {m: load_json(results / "calib" / f"{m}_D0_math500" / "summary.json")["groups"][f"{m}/D0/math500"]["mean_tokens"]
          for m in ("base", "p50c")}
    p = PLANS[name]
    n_edits = len(dr.grid((0, 1), p["ks"], p["alphas"]))      # two layer sets
    lo_hi = {"edit": (d0["p50c"], d0["base"]), "d3": (mean["base/D3/math500"], mean["p50c/D3/math500"])}
    screen_n = SCREEN_PROBLEMS[1] - SCREEN_PROBLEMS[0]
    parts = {
        "rq2-dev": (p["dev_problems"] * d0["base"],) * 2,
        "rq2-screen": tuple(screen_n * (n_edits + len(REFERENCES)) * x for x in (0.85 * SCREEN_MAX_TOKENS,
                                                                                 SCREEN_MAX_TOKENS)),
        "rq2-holdout": tuple(p["top"] * p["holdout_problems"] * x for x in lo_hi["edit"]),
        "rq2-final": tuple(sum(500 * n * (lo_hi["d3"][i] if c == "D3" else lo_hi["edit"][i]) for c, n in p["final"])
                           + len(p["controls"]) * p["control_problems"] * lo_hi["edit"][i] for i in (0, 1)),
    }
    tok = [sum(v[i] for v in parts.values()) for i in (0, 1)]
    per_gpu = min(tput)                                        # tokens/s per GPU, the slower of the two models
    engines = 1 + 2 + 2 + len(p["final"]) + len(p["controls"])
    read_h = (0.25, 0.5) if name == "full" else (0.1, 0.2)     # rough: HF teacher forcing of the windows
    hours = [tok[i] / (gpus * per_gpu * f) / 3600 + read_h[i] + engines * ENGINE_LOAD_MIN / 60
             for i, f in ((0, 1.3), (1, 0.7))]                # throughput +-30%
    return {"plan": name, "gpus": gpus, "tokens_per_s_per_gpu": per_gpu, "tokens_by_stage_low_high": parts,
            "tokens_low_high": tok, "hours_low_high": hours,
            "yuan_low_high": [h * YUAN_PER_HOUR for h in hours],
            "yuan_low_high_with_setup": [(h + SETUP_HOURS) * YUAN_PER_HOUR for h in hours],
            "assumptions": "edit samples as long as p50c's (low) to base's (high) under D0 on MATH-500; screening "
                           "samples 85-100% of the 8,192-token cap; throughput of the D0 calibration +-30%; engine "
                           f"starts {ENGINE_LOAD_MIN:g} min each; setup.sh {SETUP_HOURS:g} h if the data disk was "
                           f"wiped; {YUAN_PER_HOUR:g} yuan per hour"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate")
    s = sub.add_parser("plan")
    s.add_argument("name", choices=sorted(PLANS))
    for name in ("read", "screen", "holdout"):
        sub.add_parser(name).add_argument("--shard", help="i/k")
    sub.choices["read"].add_argument("--max-context", type=int,
                                     help="left-truncate longer contexts (only if the full context runs out of memory)")
    sub.add_parser("build")
    sub.add_parser("swapcheck").add_argument("--reference", action="store_true",
                                             help="record p50c's greedy tokens with p50c loaded from disk")
    sub.add_parser("choose").add_argument("step", choices=("screen", "holdout"))
    sub.add_parser("export")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        est = {n: estimate(n) for n in sorted(PLANS)}
        write_json(RESULTS / "cost_estimate.json", est)
        for n, e in est.items():
            print(f"{n}: {e['tokens_low_high'][0] / 1e6:.0f}-{e['tokens_low_high'][1] / 1e6:.0f}M tokens, "
                  f"{e['hours_low_high'][0]:.1f}-{e['hours_low_high'][1]:.1f} h, "
                  f"{e['yuan_low_high'][0]:.0f}-{e['yuan_low_high'][1]:.0f} yuan "
                  f"({e['yuan_low_high_with_setup'][0]:.0f}-{e['yuan_low_high_with_setup'][1]:.0f} with setup)")
        return
    if a.cmd == "plan":
        write_json(RESULTS / "plan.json", {"plan": a.name, **{k: list(v) if isinstance(v, tuple) else v
                                                             for k, v in PLANS[a.name].items()}})
        return
    {"read": cmd_read, "build": cmd_build, "swapcheck": cmd_swapcheck, "screen": cmd_screen, "holdout": cmd_holdout,
     "choose": cmd_choose, "export": cmd_export, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
