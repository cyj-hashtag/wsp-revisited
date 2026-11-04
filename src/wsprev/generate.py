"""Generation with vLLM for RQ1 / RQ2: one request per sample; every finished sample is graded and written at once.

    python -m wsprev.generate --model base --config D0 --bench math500 --samples 1 --limit 100 --out <folder>

Each sample is its own vLLM request with seed = configuration seed + sample index (decoding.Decoding.sample_seed).
With a breaker (D3, D4) the thinking is the same uninterrupted request as without one; the breaker watches it through
the engine's step loop and, when it fires, aborts the request and sends the answer request (decoding.Breaker).

Output folder (one per model x configuration x benchmark x shard):
  records.jsonl    one JSON line per finished sample, appended and flushed (fields: record_of + grade.grade)
  texts.jsonl.gz   {"problem_id", "sample_idx", "text"} of every sample (appended gzip members; never in git)
  meta.json        the job, and per invocation: command line, git commit, library versions, GPU, timing, tokens
Re-running the same command continues where it stopped: finished samples are skipped. --limit only chooses how many
problems to cover, so a later run with a larger --limit adds to the same folder.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import data, decoding, grade
from ._vendor import REPO
from .loopdetect import dup_frac

THINK_END = "</think>"
DUP_WINDOW, DUP_N = 2048, 8      # dup8_tail of every record: the last 2,048 thinking tokens


@dataclass
class Delta:
    """New output of one request since the previous engine step."""
    rid: str
    token_ids: list
    text: str
    finished: bool
    finish_reason: str | None


class VLLMEngine:
    """Step-wise access to vLLM's engine (the loop LLM.generate runs internally)."""

    def __init__(self, model_dir, gpu_memory_utilization=0.9, max_model_len=decoding.MAX_MODEL_LEN,
                 seed=decoding.SEED, extra_args=None):
        from vllm import LLM  # noqa: PLC0415

        # the extension raises vLLM's 1,024-token limit on logit_bias, which D3's carried presence penalty can exceed
        self.llm = LLM(model=str(model_dir), dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, seed=seed,
                       **{"worker_extension_cls": "wsprev._vllm_ext.LogitBiasLimit", **(extra_args or {})})
        self.engine = self.llm.llm_engine

    def add(self, rid, prompt_ids, sampling):
        from vllm import SamplingParams  # noqa: PLC0415
        from vllm.sampling_params import RequestOutputKind  # noqa: PLC0415

        params = SamplingParams(n=1, output_kind=RequestOutputKind.DELTA, **sampling)
        self.engine.add_request(rid, {"prompt_token_ids": list(prompt_ids)}, params)

    def step(self):
        out = []
        for o in self.engine.step():
            c = o.outputs[0] if o.outputs else None
            out.append(Delta(o.request_id, list(c.token_ids) if c else [], c.text if c else "", o.finished,
                             c.finish_reason if c else None))
        return out

    def abort(self, rid):
        self.engine.abort_request([rid])

    def has_unfinished(self):
        return self.engine.has_unfinished_requests()


@dataclass
class Sample:
    row: dict
    k: int                                   # sample index
    seed: int
    prompt_ids: list
    ids: list = field(default_factory=list)  # tokens of the first request (thinking, then the answer if it closed)
    texts: list = field(default_factory=list)
    close_at: int | None = None              # index of the model's own </think> in ids
    next_check: int = 0                      # next breaker checkpoint, in thinking tokens
    trigger_pos: int | None = None
    trigger_reason: str | None = None        # "dup" or "length"
    trigger_dup: float | None = None
    answer_ids: list = field(default_factory=list)
    answer_texts: list = field(default_factory=list)


class Runner:
    """Runs samples of one decoding configuration on an engine; calls on_done(sample, finish_reason) for each."""

    def __init__(self, engine, tokenizer, cfg: decoding.Decoding, on_done, log=print, log_every=60.0):
        self.engine, self.tok, self.cfg, self.on_done, self.log, self.log_every = (
            engine, tokenizer, cfg, on_done, log, log_every)
        self.br = cfg.breaker
        if self.br and not cfg.thinking:
            raise ValueError("a breaker needs thinking on")
        self.think_end = tokenizer.convert_tokens_to_ids(THINK_END) if cfg.thinking else None
        self.close_ids = list(tokenizer.encode(decoding.CLOSE_THINK, add_special_tokens=False))
        if cfg.thinking and (not isinstance(self.think_end, int) or self.think_end not in self.close_ids):
            raise ValueError(f"the tokenizer has no single {THINK_END} token; thinking cannot be tracked")
        self.active: dict[str, Sample] = {}
        self.aborted: set[str] = set()
        self.tokens = 0
        self.done = 0

    def prompt_ids(self, row) -> list:
        text = self.tok.apply_chat_template([{"role": "user", "content": data.user_message(row)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=self.cfg.thinking)
        return list(self.tok.encode(text))   # as vLLM tokenizes a text prompt (Qwen's tokenizer adds no BOS)

    def run(self, jobs) -> dict:
        """jobs: (row, sample index) pairs."""
        for row, k in jobs:
            s = Sample(row, k, self.cfg.sample_seed(k), self.prompt_ids(row))
            s.next_check = self.br.every if self.br else 0
            rid = f"{row['id']}#{k}"
            self.active[rid] = s
            self.engine.add(rid, s.prompt_ids, self.cfg.sampling(seed=s.seed))
        t0 = last = time.time()
        while self.engine.has_unfinished():
            for d in self.engine.step():
                self._on_delta(d)
            now = time.time()
            if now - last >= self.log_every:
                last = now
                self.log(f"[{(now - t0) / 60:6.1f} min] {self.done}/{len(jobs)} samples done, {len(self.active)} "
                         f"running, {self.tokens:,} tokens, {self.tokens / (now - t0):,.0f} tok/s", flush=True)
        if self.active:
            raise RuntimeError(f"the engine stopped with {len(self.active)} unfinished samples")
        seconds = time.time() - t0
        return {"samples": self.done, "tokens": self.tokens, "seconds": seconds,
                "tokens_per_s": self.tokens / seconds if seconds else None}

    # ------------------------------------------------------------ internals
    def _on_delta(self, d: Delta):
        if d.rid in self.aborted:          # tokens the engine produced before it saw our abort
            return
        s = self.active[d.rid]
        self.tokens += len(d.token_ids)
        if d.rid.endswith("/answer"):
            s.answer_ids += d.token_ids
            s.answer_texts.append(d.text)
        else:
            start = len(s.ids)
            s.ids += d.token_ids
            s.texts.append(d.text)
            if self.think_end is not None and s.close_at is None:
                close = next((start + i for i, t in enumerate(d.token_ids) if t == self.think_end), None)
                # the breaker sees the thinking up to the model's own </think>; a request that ends in this step ended
                # by itself (end of text or the token cap) and is recorded as it is
                if self.br and not d.finished and self._check(d.rid, s, len(s.ids) if close is None else close):
                    return
                s.close_at = close
        if d.finished:
            del self.active[d.rid]
            if d.finish_reason not in ("stop", "length"):   # e.g. aborted from outside: partial, do not record
                raise RuntimeError(f"{d.rid}: finish reason {d.finish_reason!r}")
            self.done += 1
            self.on_done(s, d.finish_reason)

    def _check(self, rid: str, s: Sample, limit: int) -> bool:
        """Breaker checkpoints up to `limit` thinking tokens; True if the breaker fired."""
        br = self.br
        while s.next_check <= limit:
            t = s.next_check
            dup = dup_frac(s.ids[max(0, t - br.window):t], br.ngram)
            reason = "dup" if br.tau is not None and dup >= br.tau else "length" if t >= br.max_think else None
            if reason:
                self._fire(rid, s, t, reason, dup)
                return True
            s.next_check += br.every
        return False

    def _fire(self, rid: str, s: Sample, t: int, reason: str, dup: float):
        """Cut the thinking at t tokens, close it and send the answer request."""
        self.engine.abort(rid)
        self.aborted.add(rid)
        del self.active[rid]
        s.trigger_pos, s.trigger_reason, s.trigger_dup = t, reason, dup
        s.ids = s.ids[:t]
        s.texts = [self.tok.decode(s.ids, skip_special_tokens=True)]
        override = {"max_tokens": self.br.answer_tokens, "seed": s.seed}
        if self.br.carry_presence and self.cfg.presence_penalty:
            override["presence_penalty"] = 0.0
            override["logit_bias"] = {int(x): -self.cfg.presence_penalty for x in set(s.ids) | set(self.close_ids)}
        arid = rid + "/answer"
        self.active[arid] = s
        self.engine.add(arid, s.prompt_ids + s.ids + self.close_ids, self.cfg.sampling(**override))


def record_of(s: Sample, cfg: decoding.Decoding, n_close: int, finish_reason: str, base: dict) -> tuple[dict, str]:
    """(record without grading, full generated text) of a finished sample."""
    text = "".join(s.texts)
    n_gen = len(s.ids)
    if s.trigger_pos is not None:
        text += decoding.CLOSE_THINK + "".join(s.answer_texts)
        n_gen += n_close + len(s.answer_ids)
        think_tokens = s.trigger_pos
    elif cfg.thinking:
        think_tokens = s.close_at if s.close_at is not None else len(s.ids)
    else:
        think_tokens = 0
    tail = s.ids[:think_tokens] if cfg.thinking else s.ids
    rec = dict(base)
    rec.update({
        "problem_id": s.row["id"], "sample_idx": s.k, "seed": s.seed,
        "n_gen_tokens": n_gen, "think_closed": (s.close_at is not None) if cfg.thinking else None,
        "think_tokens": think_tokens, "finish_reason": finish_reason, "cut": finish_reason == "length",
        "loop_triggered": s.trigger_pos is not None, "trigger_pos": s.trigger_pos,
        "trigger_reason": s.trigger_reason, "trigger_dup8": s.trigger_dup,
        "answer_tokens": len(s.answer_ids) if s.trigger_pos is not None else None,
        "dup8_tail": dup_frac(tail[-DUP_WINDOW:], DUP_N),
    })
    return rec, text


class Output:
    """A job's output folder (see the module docstring)."""

    def __init__(self, folder: Path, job: dict):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.records = self.folder / "records.jsonl"
        self.texts = self.folder / "texts.jsonl.gz"
        self.meta_path = self.folder / "meta.json"
        self.job = json.loads(json.dumps(job))
        self.meta = {"job": self.job, "sessions": []}
        if self.meta_path.exists():
            old = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if old["job"] != self.job:
                raise SystemExit(f"{self.folder} holds another job:\n  there: {old['job']}\n  now:   {self.job}")
            self.meta = old
        self.done = self._load_done()

    def _load_done(self) -> set:
        if not self.records.exists():
            return set()
        raw = self.records.read_bytes()
        keep = raw[:raw.rfind(b"\n") + 1]      # drop a line cut off by a crash
        if len(keep) != len(raw):
            self.records.write_bytes(keep)
        return {(r["problem_id"], r["sample_idx"]) for r in map(json.loads, keep.decode("utf-8").splitlines())}

    def write(self, rec: dict, text: str):
        with gzip.open(self.texts, "at", encoding="utf-8") as g:
            g.write(json.dumps({"problem_id": rec["problem_id"], "sample_idx": rec["sample_idx"], "text": text},
                               ensure_ascii=False) + "\n")
        with open(self.records, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done.add((rec["problem_id"], rec["sample_idx"]))

    def start_session(self, info: dict) -> dict:
        session = dict(info, start=time.strftime("%Y-%m-%d %H:%M:%S %z"), end=None)
        self.meta["sessions"].append(session)
        self._save()
        return session

    def end_session(self, session: dict, stats: dict):
        session.update(stats, end=time.strftime("%Y-%m-%d %H:%M:%S %z"))
        self._save()

    def _save(self):
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.meta_path)


# ---------------------------------------------------------------- environment facts for meta.json
def git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def versions() -> dict:
    from importlib import metadata  # noqa: PLC0415

    out = {"python": platform.python_version()}
    for pkg in ("vllm", "transformers", "torch", "math-verify", "numpy", "nltk", "langdetect"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def gpu_info() -> dict:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count()}
    except ImportError:
        pass
    return {"name": None, "count": 0}


def resolve_model(name_or_path: str) -> tuple[str, Path, dict]:
    """A model folder made by scripts/setup.sh ("base", "p50c" under $WSPREV_MODELS) or a path; its
    wsprev_model.json gives the name and the weight file's sha256."""
    p = Path(name_or_path)
    if not p.exists():
        p = Path(os.environ.get("WSPREV_MODELS", data.home() / "models")) / name_or_path
    if not p.exists():
        raise SystemExit(f"model {name_or_path!r} not found (looked at {p})")
    info_path = p / "wsprev_model.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return info.get("name", p.name), p.resolve(), info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base, p50c (folders under $WSPREV_MODELS) or a model folder")
    ap.add_argument("--config", required=True, help="D0, D1, D2a, D2b, D3, D3@<tau>, D4 (wsprev.decoding)")
    ap.add_argument("--bench", required=True, help="math500, gsm8k, ifeval, practice:dev|holdout|select")
    ap.add_argument("--samples", type=int, help="samples per problem (default: RQ1's count, 1 for practice)")
    ap.add_argument("--limit", type=int, help="only the first LIMIT problems")
    ap.add_argument("--shard", help="i/k: problems i, i+k, i+2k, ... (after --limit)")
    ap.add_argument("--out", help="output folder (default: $WSPREV_RUNS/<model>_<config>_<bench>[_shard])")
    ap.add_argument("--gpu-mem", type=float, default=0.9, help="vLLM gpu_memory_utilization")
    ap.add_argument("--vllm-arg", action="append", default=[], metavar="KEY=JSON",
                    help="extra vllm.LLM argument, e.g. enable_prefix_caching=true (repeatable)")
    ap.add_argument("--log-every", type=float, default=60.0, help="seconds between progress lines")
    a = ap.parse_args(argv)

    cfg = decoding.get(a.config)
    rows = data.load(a.bench)
    if a.limit:
        rows = rows[:a.limit]
    shard_tag = ""
    if a.shard:
        i, k = map(int, a.shard.split("/"))
        rows = rows[i::k]
        shard_tag = f"_s{i}of{k}"
    n = a.samples or (data.RQ1_SAMPLES.get(a.bench, 1))
    name, model_dir, model_info = resolve_model(a.model)
    folder = Path(a.out) if a.out else (Path(os.environ.get("WSPREV_RUNS", data.home() / "runs"))
                                        / f"{name}_{cfg.name}_{a.bench.replace(':', '-')}{shard_tag}")
    job = {"model": name, "model_dir": str(model_dir), "model_sha": model_info.get("weight_sha256"),
           "config": cfg.to_dict(), "benchmark": a.bench, "samples": n, "shard": a.shard}
    out = Output(folder, job)
    jobs = [(r, k) for r in rows for k in range(n) if (r["id"], k) not in out.done]
    print(f"{name} {cfg.name} {a.bench}: {len(jobs)} samples to generate, {len(out.done)} already in {folder}",
          flush=True)
    if not jobs:
        return

    vllm_args = {}
    for kv in a.vllm_arg:
        key, val = kv.split("=", 1)
        vllm_args[key] = json.loads(val)
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_dir)
    engine = VLLMEngine(model_dir, a.gpu_mem, extra_args=vllm_args)
    session = out.start_session({"argv": sys.argv, "git": git_state(), "versions": versions(), "gpu": gpu_info(),
                                 "host": platform.node(), "vllm_args": vllm_args, "model_info": model_info,
                                 "samples_requested": len(jobs)})
    base = {"benchmark": a.bench, "model": name, "model_sha": model_info.get("weight_sha256"), "config": cfg.name}
    n_close = len(tok.encode(decoding.CLOSE_THINK, add_special_tokens=False))

    def on_done(s, reason):
        rec, text = record_of(s, cfg, n_close, reason, base)
        try:
            rec.update(grade.grade(s.row, text, cfg.thinking))
        except Exception as e:  # noqa: BLE001 - keep the generation; regrade later from texts.jsonl.gz
            rec.update({"pred": None, "correct": False, "grade_error": repr(e)[:300]})
        out.write(rec, text)

    stats = {}
    try:
        stats = Runner(engine, tok, cfg, on_done, log_every=a.log_every).run(jobs)
    finally:
        out.end_session(session, stats)
    print(f"done: {stats['samples']} samples, {stats['tokens']:,} tokens in {stats['seconds'] / 60:.1f} min "
          f"({stats['tokens_per_s']:,.0f} tok/s) -> {folder}", flush=True)


if __name__ == "__main__":
    main()
