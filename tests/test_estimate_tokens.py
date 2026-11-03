"""scripts/estimate_tokens.py replays the breaker exactly as generate.Runner runs it (same streams as test_generate)."""
from test_generate import BR, oracle, stream
from test_scripts import script

from fakes import THINK_END


def test_replay_matches_the_runner():
    est = script("estimate_tokens")
    ids = stream("loop", 400)
    assert est.first_fire(est.checkpoints(ids, None, BR), BR.tau, BR.max_think) == oracle("loop", BR)
    fresh = stream("fresh", 400)
    assert est.first_fire(est.checkpoints(fresh, None, BR), BR.tau, BR.max_think) == BR.max_think


def test_replay_respects_the_models_own_close_and_the_end_of_a_request():
    est = script("estimate_tokens")
    closed = list(range(1000, 1030)) + [THINK_END, 7, 1]       # thinking closes after 30 tokens
    assert est.checkpoints(closed, 30, BR) == [(16, 0.0)]       # only checkpoint 16 is reached before </think>
    assert est.first_fire(est.checkpoints(closed, 30, BR), BR.tau, BR.max_think) is None
    ended = list(range(1000, 1016))                             # ends (no </think>) in the step that reaches 16
    assert est.checkpoints(ended, None, BR) == []
