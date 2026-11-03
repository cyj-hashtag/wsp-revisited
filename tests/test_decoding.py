"""The configurations match the plan's table (docs/plan-for-agents.md, section 3) and the original think35."""
import pytest

from wsprev import decoding

THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768}  # [B] evaluate.py


@pytest.mark.parametrize("name, thinking, t, p, k, pp", [
    ("D0", True, 0.6, 0.95, 20, 1.5),
    ("D1", True, 1.0, 0.95, 20, 1.5),
    ("D2a", False, 1.0, 1.00, 20, 2.0),
    ("D2b", False, 0.7, 0.80, 20, 1.5),
])
def test_table(name, thinking, t, p, k, pp):
    c = decoding.get(name)
    assert (c.thinking, c.temperature, c.top_p, c.top_k, c.presence_penalty) == (thinking, t, p, k, pp)
    assert (c.min_p, c.repetition_penalty, c.max_tokens, c.seed, c.breaker) == (0.0, 1.0, 32768, 0, None)


def test_d0_is_think35():
    sp = decoding.D0.sampling()
    assert {k: sp[k] for k in THINK35} == THINK35 and decoding.D0.thinking


def test_d3_needs_calibrated_tau(monkeypatch):
    assert decoding.D3_TAU is None or decoding.D3_TAU in decoding.TAU_CANDIDATES
    monkeypatch.setattr(decoding, "D3_TAU", None)
    with pytest.raises(ValueError):
        decoding.get("D3")
    monkeypatch.setattr(decoding, "D3_TAU", 0.7)
    assert decoding.get("D3").name == "D3" and decoding.get("D3").breaker.tau == 0.7
    c = decoding.get("D3@0.5")
    b = c.breaker
    assert (b.tau, b.max_think, b.every, b.window, b.ngram, b.answer_tokens) == (0.5, 30720, 1024, 2048, 8, 2048)
    assert c.sampling() == decoding.D0.sampling() and c.thinking
    assert 30720 % b.every == 0


def test_d4():
    b = decoding.get("D4").breaker
    assert b.tau is None and b.max_think == 8192 and decoding.get("D4").sampling() == decoding.D0.sampling()


def test_lengths():
    assert decoding.MAX_MODEL_LEN == 32768 + 2048
    assert decoding.CLOSE_THINK == "\n</think>\n\n"
    with pytest.raises(KeyError):
        decoding.get("D9")
