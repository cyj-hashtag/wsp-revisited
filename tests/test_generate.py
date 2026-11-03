"""generate.Runner / record_of / Output with a scripted engine (no GPU, no vLLM)."""
import itertools
import json
from dataclasses import replace

import pytest
from fakes import BOX, EOS, NL, NL2, THINK_END, FakeEngine, FakeTok

from wsprev import decoding
from wsprev.decoding import Breaker
from wsprev.generate import Output, Runner, record_of
from wsprev.loopdetect import dup_frac

BR = Breaker(tau=0.5, max_think=192, every=16, window=32, ngram=4, answer_tokens=20)
CFG = replace(decoding.D0, name="T3", breaker=BR)


def script(pid, seed, is_answer):  # noqa: ARG001
    if is_answer:
        return iter([BOX, EOS])
    if pid == "loop":      # 40 distinct tokens, then a 5-token cycle forever
        return itertools.chain(range(1000, 1040), itertools.cycle(range(2000, 2005)))
    if pid == "close":     # closes the thinking by itself after 30 tokens, answers, ends
        return iter([*range(1000, 1030), THINK_END, BOX, BOX, BOX, EOS])
    if pid == "fresh":     # never repeats: only the length limit can stop it
        return itertools.count(3000)
    if pid == "eos16":     # ends (end of text) exactly at the first checkpoint
        return iter([*range(1000, 1015), EOS])
    raise KeyError(pid)


def stream(pid, n):
    return list(itertools.islice(script(pid, 0, False), n))


def oracle(pid, br):
    """First checkpoint where the breaker fires, computed directly from the definition."""
    ids = stream(pid, br.max_think + br.every)
    for t in range(br.every, br.max_think + br.every, br.every):
        if (br.tau is not None and dup_frac(ids[max(0, t - br.window):t], br.ngram) >= br.tau) or t >= br.max_think:
            return t
    raise AssertionError


def run(pids, cfg=CFG, **engine_kw):
    engine, done = FakeEngine(script, **engine_kw), {}
    runner = Runner(engine, FakeTok(), cfg, lambda s, r: done.__setitem__(s.row["id"], (s, r)), log=lambda *a, **k: None)
    runner.run([({"id": p, "problem": p, "answer": "7"}, 0) for p in pids])
    recs = {p: record_of(s, cfg, 3, r, {"config": cfg.name}) for p, (s, r) in done.items()}
    return engine, done, recs


@pytest.mark.parametrize("per_step", [1, 3])
@pytest.mark.parametrize("late", [False, True])
def test_loop_breaker_fires_at_first_checkpoint_over_tau(per_step, late):
    engine, done, recs = run(["loop"], per_step=per_step, late=late)
    t = oracle("loop", BR)
    s, reason = done["loop"]
    rec, text = recs["loop"]
    assert (s.trigger_pos, s.trigger_reason, reason) == (t, "dup", "stop")
    assert t % BR.every == 0 and t < BR.max_think
    assert s.ids == stream("loop", t)                              # thinking cut exactly at the checkpoint
    assert engine.aborted == ["loop#0"]
    rid, prompt, sp = engine.added[-1]
    assert rid == "loop#0/answer"
    assert prompt == engine.added[0][1] + stream("loop", t) + [NL, THINK_END, NL2]
    assert sp["max_tokens"] == BR.answer_tokens and sp["seed"] == 0
    assert sp["presence_penalty"] == 0.0                           # carry_presence
    assert sp["logit_bias"] == {x: -1.5 for x in set(stream("loop", t)) | {NL, THINK_END, NL2}}
    assert rec["loop_triggered"] and rec["trigger_pos"] == t and rec["think_tokens"] == t
    assert rec["think_closed"] is False and rec["answer_tokens"] == 2
    assert rec["n_gen_tokens"] == t + 3 + 2
    assert rec["trigger_dup8"] == pytest.approx(dup_frac(stream("loop", t)[t - BR.window:t], BR.ngram))
    assert text.endswith("\n</think>\n\n\\boxed{7}") and "<late>" not in text


def test_without_carry_presence_answer_keeps_the_sampling():
    cfg = replace(CFG, breaker=replace(BR, carry_presence=False))
    engine, _, _ = run(["loop"], cfg=cfg)
    sp = engine.added[-1][2]
    assert "logit_bias" not in sp and sp["presence_penalty"] == 1.5


def test_natural_close_is_not_touched():
    engine, done, recs = run(["close"])
    rec, text = recs["close"]
    assert not rec["loop_triggered"] and rec["think_closed"] and rec["think_tokens"] == 30
    assert rec["n_gen_tokens"] == 35 and rec["finish_reason"] == "stop" and not rec["cut"]
    assert engine.aborted == [] and len(engine.added) == 1
    assert text.endswith("</think>\\boxed{7}\\boxed{7}\\boxed{7}")


def test_length_trigger():
    _, done, recs = run(["fresh"])
    s, _ = done["fresh"]
    assert (s.trigger_pos, s.trigger_reason) == (BR.max_think, "length") == (oracle("fresh", BR), "length")


def test_request_ending_at_a_checkpoint_is_recorded_as_is():
    engine, _, recs = run(["eos16"])
    rec, _ = recs["eos16"]
    assert not rec["loop_triggered"] and rec["finish_reason"] == "stop" and engine.aborted == []


def test_plain_config_runs_to_the_cap():
    cfg = replace(decoding.D0, max_tokens=100)
    engine, _, recs = run(["loop", "close"], cfg=cfg)
    rec, _ = recs["loop"]
    assert rec["cut"] and rec["n_gen_tokens"] == 100 and not rec["loop_triggered"] and rec["think_tokens"] == 100
    assert rec["dup8_tail"] > 0.5 and engine.aborted == []
    assert {r for r, _, _ in engine.added} == {"loop#0", "close#0"}
    assert all(sp["seed"] == 0 and sp["presence_penalty"] == 1.5 for _, _, sp in engine.added)


def test_non_thinking_config():
    cfg = replace(decoding.D2b, max_tokens=50)
    engine, _, recs = run(["close"], cfg=cfg)
    rec, _ = recs["close"]
    assert rec["think_closed"] is None and rec["think_tokens"] == 0
    assert engine.added[0][1] == FakeTok().encode("<u>close\nPlease reason step by step, and put your final answer "
                                                  "within \\boxed{}.<think>\n\n</think>\n\n")


def test_sample_seeds():
    engine, done = FakeEngine(script), []
    Runner(engine, FakeTok(), decoding.D0, lambda s, r: done.append(s), log=lambda *a, **k: None).run(
        [({"id": "close", "problem": "x", "answer": "7"}, k) for k in range(3)])
    assert sorted(sp["seed"] for _, _, sp in engine.added) == [0, 1, 2]


def test_output_resume_and_job_check(tmp_path):
    job = {"model": "base", "config": {"name": "D0"}, "benchmark": "math500", "samples": 1, "shard": None}
    out = Output(tmp_path, job)
    out.write({"problem_id": "a", "sample_idx": 0, "correct": True}, "text a")
    with open(out.records, "a", encoding="utf-8") as f:
        f.write('{"problem_id": "b", "sampl')                      # a crash in the middle of a line
    again = Output(tmp_path, job)
    assert again.done == {("a", 0)}
    assert again.records.read_text(encoding="utf-8").count("\n") == 1
    s = again.start_session({"argv": ["x"]})
    again.end_session(s, {"samples": 0})
    assert json.loads(again.meta_path.read_text(encoding="utf-8"))["sessions"][0]["samples"] == 0
    with pytest.raises(SystemExit):
        Output(tmp_path, dict(job, samples=2))
