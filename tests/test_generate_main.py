"""The command line path of wsprev.generate and scripts/summarize.py, with vLLM and the tokenizer replaced by fakes."""
import itertools
import json
import sys
import types

from fakes import BOX, EOS, THINK_END, FakeEngine, FakeTok
from test_scripts import script

from wsprev import generate


def behaviour(pid, seed, is_answer):  # noqa: ARG001
    if is_answer:
        return iter([BOX, EOS])
    if int(pid[1:]) % 2:   # odd problems loop
        return itertools.chain(range(1000, 1040), itertools.cycle(range(2000, 2005)))
    return iter([*range(1000, 1030), THINK_END, BOX, EOS])


def test_generate_main_then_summarize(tmp_path, monkeypatch, capsys):
    bench = tmp_path / "bench"
    bench.mkdir()
    with open(bench / "math500.jsonl", "w", encoding="utf-8") as f:
        for i in range(500):
            f.write(json.dumps({"id": f"p{i}", "problem": f"problem {i}", "answer": "7"}) + "\n")
    model = tmp_path / "models" / "base"
    model.mkdir(parents=True)
    (model / "wsprev_model.json").write_text(json.dumps({"name": "base", "weight_sha256": "abc"}))
    monkeypatch.setenv("WSPREV_BENCH", str(bench))
    monkeypatch.setenv("WSPREV_MODELS", str(tmp_path / "models"))
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: FakeTok())
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(generate, "VLLMEngine", lambda *a, **k: FakeEngine(behaviour, per_step=4))

    out = tmp_path / "out"
    args = ["--model", "base", "--config", "D3@0.5", "--bench", "math500", "--samples", "2", "--limit", "6",
            "--out", str(out)]
    generate.main(args)
    recs = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 12 and {(r["problem_id"], r["sample_idx"]) for r in recs} == {
        (f"p{i}", k) for i in range(6) for k in range(2)}
    for r in recs:
        odd = int(r["problem_id"][1:]) % 2
        assert r["loop_triggered"] == bool(odd) and r["pred"] == "7" and r["model_sha"] == "abc"
        assert r["config"] == "D3@0.5" and r["seed"] == r["sample_idx"]
        if odd:
            assert r["trigger_pos"] == 1024 and r["trigger_reason"] == "dup"
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["job"]["config"]["breaker"]["tau"] == 0.5 and meta["sessions"][0]["samples"] == 12

    generate.main(args)                               # resumes: nothing left to do
    assert "0 samples to generate, 12 already" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["summarize.py", str(out), "--out", str(tmp_path / "summary")])
    script("summarize").main()
    summary = json.loads((tmp_path / "summary" / "summary.json").read_text(encoding="utf-8"))
    g = summary["groups"]["base/D3@0.5/math500"]
    assert g["problems"] == 6 and g["samples"] == 12 and g["breaker_share"] == 0.5 and g["cut_share"] == 0.0
