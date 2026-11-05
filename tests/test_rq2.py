"""RQ2 pipeline (src/wsprev/rq2.py) on the CPU: model folders, build, the selection rules, the controls, the swap
runner with a fake engine, and the RQ2 stages of scripts/run_stage.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fakes import BOX, EOS, THINK_END, FakeEngine, FakeTok
from test_scripts import script

from wsprev import directions as dr
from wsprev import rq2
from wsprev._vendor import REPO

sys.path.insert(0, str(REPO / "third_party" / "wsp_logs"))
import st_numpy as S  # noqa: E402

D, F = 8, rq2.D_FF   # tiny d_model; d_ff must be 3584 (masks and k = d_ff)


def fake_model(folder: Path, layers=(15, 19), seed=0) -> dict:
    """A model folder with bfloat16 down_proj tensors of shape (D, F) for `layers` and one other tensor."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {dr.TENSOR.format(l): ("BF16", S.bf16_bits(rng.standard_normal((D, F)).astype(np.float32)))
               for l in layers}
    tensors["model.language_model.embed_tokens.weight"] = ("BF16", S.bf16_bits(np.ones((4, D), np.float32)))
    S.save(str(folder / rq2.WEIGHTS), tensors, {"format": "pt"})
    (folder / "config.json").write_text('{"x": 1}')
    return tensors


def test_write_model_changes_only_the_given_tensors(tmp_path):
    t = fake_model(tmp_path / "base")
    name = dr.TENSOR.format(15)
    new = t[name][1].copy()
    new[0, :5] ^= 1
    meta = rq2.write_model(tmp_path / "base", {name: new}, tmp_path / "out", {"name": "x", "edit": {"k": 1}})
    out = S.load(str(tmp_path / "out" / rq2.WEIGHTS))
    assert np.array_equal(out[name][1], new)
    assert np.array_equal(out[dr.TENSOR.format(19)][1], t[dr.TENSOR.format(19)][1])
    assert meta["changed_entries"] == {name: 5} and meta["edit"] == {"k": 1}
    assert json.loads((tmp_path / "out" / "wsprev_model.json").read_text())["weight_sha256"] == meta["weight_sha256"]
    assert (tmp_path / "out" / "config.json").read_text() == '{"x": 1}'
    with pytest.raises(ValueError):
        rq2.write_model(tmp_path / "base", {name: new[:, :10]}, tmp_path / "bad", {"name": "bad"})


def fake_read(pairs=12, seed=1, layers=dr.LAYERS):
    """Window means where layer 19's loop windows are shifted along a planted unit vector."""
    rng = np.random.default_rng(seed)
    L = len(layers)
    v = np.zeros(D)
    v[0] = 1.0
    res, act, kind, pair = [], [], [], []
    for p in range(pairs):
        for k in ("loop", "normal"):
            r = rng.standard_normal((L, D))
            a = rng.random((L, F))
            if k == "loop":
                r[layers.index(19)] += 4 * v
                a[:, :3] += 1.0        # neurons 0-2 fire more in loops
            res.append(r), act.append(a), kind.append(k), pair.append(p)
    res.append(rng.standard_normal((L, D))), act.append(rng.random((L, F))), kind.append("finished"), pair.append(-1)
    n = len(kind)
    return {"res": np.array(res), "act": np.array(act), "kind": np.array(kind), "pair": np.array(pair),
            "problem_id": np.array([f"p{i}" for i in range(n)]), "dup8": np.zeros(n), "context": np.ones(n),
            "layers": np.array(layers)}


def weights(layers=dr.LAYERS, seed=2):
    rng = np.random.default_rng(seed)
    W = {l: rng.standard_normal((D, F)).astype(np.float32) for l in layers}
    for l in layers:
        W[l][0, :3] = 5.0               # neurons 0-2 write along the planted direction
    return W


PLAN = {"name": "lean", **rq2.PLANS["lean"]}


def test_build_and_edit_weights():
    W = weights()
    b = rq2.build(fake_read(), W, PLAN)
    assert 19 in b["auc_layers"] and b["layers"]["19"]["auc"] > 0.9
    assert len(b["edits"]) == (12 if tuple(b["auc_layers"]) != dr.ASTRA_LAYERS else 6)
    assert set(b["top_columns"]["19"]["64"][:3]) == {0, 1, 2}        # the planted neurons score highest
    e = rq2.edit_by_name(b, "astra-L15-19-k64-a6")
    ws = rq2.edit_weights(b, e, W)
    m = rq2.edit_mask(b, e, 19)
    assert m.sum() == 64 and set(ws) == {15, 19}
    u = np.array(b["u"]["19"])
    np.testing.assert_allclose(ws[19], dr.edited({19: W[19]}, {19: u}, {19: m}, 6.0)[19])
    full = rq2.edit_by_name(b, "astra-L15-19-k3584-a1")
    assert rq2.edit_mask(b, full, 15).sum() == F
    for chk in b["prop3"]["astra-L15-19-k64-a6"].values():
        assert chk["change"] == pytest.approx(chk["predicted_change"], rel=1e-6, abs=1e-6)


def write_records(folder: Path, n, correct, finished, model="x"):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "records.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"model": model, "config": "D0-8k", "benchmark": "b", "problem_id": f"q{i}",
                                "sample_idx": 0, "correct": i < correct, "cut": i >= finished,
                                "n_gen_tokens": 100, "think_closed": i < finished, "finish_reason": "stop",
                                "dup8_tail": 0.0}) + "\n")


def test_choose_follows_the_rules(tmp_path, monkeypatch):
    b = {"edits": [dr.Edit(f"e{i}", s, (15, 19), k, a).to_dict()
                   for i, (s, k, a) in enumerate([("astra", 64, 6), ("astra", 64, 1), ("auc", 32, 1),
                                                  ("astra", 3584, 2)])]}
    n = rq2.SCREEN_PROBLEMS[1] - rq2.SCREEN_PROBLEMS[0]
    scores = {"e0": (30, 50), "e1": (30, 50), "e2": (30, 50), "e3": (31, 40), "base": (5, 3), "p50c": (40, 55)}
    for name, (c, f) in scores.items():
        write_records(tmp_path / "screen" / name, n, c, f, name)
    s = rq2.choose("screen", b, dict(PLAN, top=3), root=tmp_path)
    # e3 has the most correct; e0-e2 tie on correct and finished: smaller alpha, then smaller k, then astra first
    assert s["ranking"] == ["e3", "e2", "e1", "e0"] and s["top"] == ["e3", "e2", "e1"]
    assert "p50c" in s["table"] and "p50c" not in s["ranking"]          # references are never chosen
    monkeypatch.setattr(rq2, "RESULTS", tmp_path / "res")
    rq2.write_json(tmp_path / "res" / "screen.json", s)
    for name, (c, f) in {"e3": (100, 120), "e2": (101, 110), "e1": (101, 110)}.items():
        write_records(tmp_path / "holdout" / name / "s0of1", PLAN["holdout_problems"], c, f, name)
    h = rq2.choose("holdout", b, PLAN, root=tmp_path)
    assert h["ranking"] == ["e2", "e1", "e3"] and h["choice"] == "e2" and h["edit"]["name"] == "e2"
    write_records(tmp_path / "holdout" / "e1" / "s0of1", 3, 1, 1, "e1")
    with pytest.raises(SystemExit, match="expected"):
        rq2.choose("holdout", b, PLAN, root=tmp_path)


def test_controls():
    W = {l: S.bf16_to_f32(S.bf16_bits(w)) for l, w in weights(layers=(15, 19)).items()}   # exact in bf16, as loaded
    b = rq2.build(fake_read(), weights(), PLAN)
    e = rq2.edit_by_name(b, "astra-L15-19-k64-a6")
    rng = np.random.default_rng(7)
    u = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    m = np.zeros(F)
    m[rng.choice(F, 64, replace=False)] = 1
    W_p50c = {l: S.bf16_to_f32(S.bf16_bits(dr.edited({l: W[l]}, {l: u}, {l: m}, 6.0)[l])) for l in (15, 19)}
    out = rq2.controls(b, e, W, W_p50c)
    assert set(out) == {"rq2-ctrl-a", "rq2-ctrl-b", "rq2-ctrl-c"}
    wa, _ = out["rq2-ctrl-a"]
    changed = np.abs(wa[19] - W[19]).sum(axis=0) > 0
    assert (changed == (rq2.edit_mask(b, e, 19) > 0)).all()             # a: the edit's columns
    wb, _ = out["rq2-ctrl-b"]
    assert (np.abs(wb[15] - W[15]).sum(axis=0) > 0).sum() == 64          # b: 64 random columns
    wc, info = out["rq2-ctrl-c"]
    assert info["alpha"] == 1.0
    for l in ("15", "19"):
        assert info["recovery_check"][l]["columns"] == 64
        assert info["recovery_check"][l]["alpha6_rebuild_equal_share"] > 0.99
    full = rq2.edit_by_name(b, "astra-L15-19-k3584-a1")
    assert "rq2-ctrl-b" not in rq2.controls(b, full, W, W_p50c)          # identical to the edit: skipped


class FakeLLM:
    def __init__(self):
        self.resets = 0

    def reset_prefix_cache(self):
        self.resets += 1


def test_run_edits_swaps_weights_per_edit(tmp_path):
    def behaviour(pid, seed, is_answer):  # noqa: ARG001
        return iter([*range(1000, 1010), THINK_END, BOX, EOS])

    engine = FakeEngine(behaviour, per_step=3)
    engine.llm = FakeLLM()
    calls, received = [], {}
    bits = np.arange(6, dtype=np.uint16).reshape(2, 3)

    def rpc(method, *args):
        # vLLM's RPC only carries plain values intact (a tensor arrived as a list on 2026-11-06): JSON must do
        json.dumps(args)
        calls.append((method, sorted(args[0]) if args else None))
        if method == "wsprev_set_down_proj":
            received.update({int(l): np.load(p) for l, p in args[0].items()})   # the files exist during the call
        return [[]]

    rows = [{"id": f"q{i}", "problem": f"problem {i}", "answer": "7"} for i in range(3)]
    edits = [("base", None, {"reference": "base"}), ("e1", {15: bits}, {"name": "e1"})]
    from wsprev import decoding

    rq2.run_edits(engine, FakeTok(), edits, rows, decoding.D0, tmp_path, None, rpc, log=lambda *a, **k: None)
    assert calls == [("wsprev_restore", None), ("wsprev_restore", None), ("wsprev_set_down_proj", [15]),
                     ("wsprev_restore", None)]
    assert np.array_equal(received[15], bits) and received[15].dtype == np.uint16
    assert not list((tmp_path / "_swap").glob("*.npy"))                 # the files are removed afterwards
    assert engine.llm.resets == 2
    for name in ("base", "e1"):
        recs = [json.loads(x) for x in (tmp_path / name / "records.jsonl").read_text().splitlines()]
        # pred, not correct: math-verify's timeouts fail under pytest on Windows (see tests/test_grade.py)
        assert len(recs) == 3 and all(r["pred"] == "7" and r["model"] == name for r in recs)
    calls.clear()
    rq2.run_edits(engine, FakeTok(), edits, rows, decoding.D0, tmp_path, None, rpc, log=lambda *a, **k: None)
    assert calls == [("wsprev_restore", None)]                          # nothing left to do: no swap


def test_paired_delta():
    g = {("a", "D0", "m"): [{"problem_id": f"p{i}", "correct": i < 6} for i in range(10)],
         ("b", "D0", "m"): [{"problem_id": f"p{i}", "correct": i < 2} for i in range(10)]}
    d = rq2.paired_delta(g, ("a", "D0", "m"), ("b", "D0", "m"), n_boot=500)
    assert d["estimate"] == pytest.approx(0.4) and d["problems"] == 10 and d["ci95"][0] <= 0.4 <= d["ci95"][1]
    assert rq2.paired_delta(g, ("a", "D0", "m"), ("c", "D0", "m")) is None


def test_plan_command(tmp_path, monkeypatch):
    monkeypatch.setattr(rq2, "RESULTS", tmp_path)
    rq2.main(["plan", "lean"])
    assert json.loads((tmp_path / "plan.json").read_text())["plan"] == "lean"
    assert rq2.plan(tmp_path)["dev_problems"] == 160
    with pytest.raises(SystemExit):
        rq2.plan(tmp_path / "none")


def test_run_stage_rq2(tmp_path, monkeypatch):
    rs = script("run_stage")
    (tmp_path / "prereg").mkdir()
    assert any("rq2_rules" in m for m in rs.missing("rq2-dev", tmp_path))
    (tmp_path / "prereg" / "rq2_rules.md").write_text("rules")
    assert any("plan.json" in m for m in rs.missing("rq2-dev", tmp_path))
    (tmp_path / "results" / "rq2").mkdir(parents=True)
    (tmp_path / "results" / "rq2" / "plan.json").write_text('{"plan": "full"}')
    assert rs.missing("rq2-dev", tmp_path) == [] and rs.missing("rq2-read", tmp_path) == []
    assert any("build.json" in m for m in rs.missing("rq2-screen", tmp_path))
    monkeypatch.setattr(rs.rq2, "RESULTS", tmp_path / "results" / "rq2")
    assert [(j.name, j.limit, j.group) for j in rs.jobs("rq2-dev")] == [("base_D0_practice-dev", 448, "rq2")]
    final = rs.jobs("rq2-final")
    assert [j.name for j in final] == ["rq2-auto_D0_math500", "rq2-auto_D3_math500", "rq2-ctrl-a_D0_math500",
                                       "rq2-ctrl-b_D0_math500", "rq2-ctrl-c_D0_math500"]
    assert {j.samples for j in final if j.config == "D3"} == {2}
    pre, par, post = rs.rq2_commands("rq2-screen", ["0", "1"])
    assert [c[1][-2:] for c in pre] == [["swapcheck", "--reference"], ["wsprev.rq2", "swapcheck"]]
    assert [c[0] for c in pre] == ["0", "0"] and [c[0] for c in par] == ["0", "1"]
    assert par[1][1][-3:] == ["screen", "--shard", "1/2"] and post[0][1][-2:] == ["choose", "screen"]
    assert post[0][0] is None


def test_report_on_fake_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(rq2, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(rq2, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    rq2.write_json(tmp_path / "results" / "plan.json", {"plan": "lean"})
    runs = tmp_path / "runs"
    monkeypatch.setenv("WSPREV_RUNS", str(runs))

    def recs(folder, model, correct):
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "records.jsonl", "w") as f:
            for i in range(10):
                f.write(json.dumps({"model": model, "config": "D0", "benchmark": "math500", "problem_id": f"p{i}",
                                    "sample_idx": 0, "correct": i < correct, "cut": False, "n_gen_tokens": 10,
                                    "think_closed": True, "finish_reason": "stop", "dup8_tail": 0.0}) + "\n")
        (folder / "meta.json").write_text(json.dumps({"job": {}, "sessions": [{"seconds": 3600, "tokens": 1000}]}))

    recs(runs / "d0" / "base", "base", 1)
    recs(runs / "d0" / "p50c", "p50c", 5)
    recs(runs / "rq2" / "final" / "auto", "rq2-auto", 4)
    rq2.main(["report"])
    out = json.loads((tmp_path / "results" / "rq2.json").read_text())
    assert out["comparisons"]["rq2-auto - base (D0)"]["estimate"] == pytest.approx(0.3)
    assert out["comparisons"]["rq2-auto - p50c (D0)"]["estimate"] == pytest.approx(-0.1)
    assert out["cost"] == {"llm_tokens": 0, "gpu_seconds": 3600, "generated_tokens": 1000}
    assert "0 LLM tokens" in (tmp_path / "results" / "rq2_summary.md").read_text()


def test_dev_records_order(tmp_path):
    import gzip

    folder = tmp_path / "dev" / "s0of1"
    folder.mkdir(parents=True)
    with open(folder / "records.jsonl", "w") as f:
        for pid in ("b", "a", "z"):
            f.write(json.dumps({"model": "base", "config": "D0", "benchmark": "practice:dev", "problem_id": pid,
                                "sample_idx": 0}) + "\n")
    with gzip.open(folder / "texts.jsonl.gz", "wt") as g:
        for pid in ("b", "a", "z"):
            g.write(json.dumps({"problem_id": pid, "sample_idx": 0, "text": pid * 3}) + "\n")
    recs, texts = rq2.dev_records(tmp_path / "dev", ["a", "b"])
    assert [r["problem_id"] for r in recs] == ["a", "b"] and texts[("z", 0)] == "zzz"




def test_estimate_uses_the_measured_rq1_numbers():
    lean, full = rq2.estimate("lean"), rq2.estimate("full")
    for e in (lean, full):
        assert 0 < e["tokens_low_high"][0] <= e["tokens_low_high"][1]
        assert 0 < e["hours_low_high"][0] < e["hours_low_high"][1]
        assert e["yuan_low_high_with_setup"][0] == pytest.approx(e["yuan_low_high"][0] + rq2.SETUP_HOURS * 3.0)
    assert lean["tokens_low_high"][1] < full["tokens_low_high"][0]
    assert lean["tokens_by_stage_low_high"]["rq2-dev"][0] == pytest.approx(160 * 30721.256)


def test_set_weights_refuses_non_bits(tmp_path):
    with pytest.raises(TypeError):
        rq2.set_weights(lambda *a: None, {15: np.zeros((2, 2), np.float32)}, tmp_path)


def test_bits_sha256_matches_the_file_bytes():
    bits = S.bf16_bits(np.array([[1.0, -2.5], [3.0, 0.0]], np.float32))
    import hashlib

    assert rq2.bits_sha256(bits) == hashlib.sha256(bits.tobytes()).hexdigest()
