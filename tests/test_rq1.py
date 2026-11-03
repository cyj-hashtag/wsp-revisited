"""The registered RQ1 analysis (prereg/criteria.md section 5) on synthetic per-problem accuracies."""
import numpy as np
import pytest

from wsprev import rq1

N = 500


def vec(k, n=N):
    """k problems solved (the first k), as per-problem accuracy."""
    v = np.zeros(n)
    v[:k] = 1.0
    return v


def cells(d0=(40, 240), d1=(50, 150), d2b=(100, 300), d3=(230, 235)):
    """(base, p50c) solved counts per configuration; p50c solves a superset of base's problems."""
    out = {}
    for c, (b, p) in {"D0": d0, "D1": d1, "D2b": d2b, "D3": d3}.items():
        out[("base", c)], out[("p50c", c)] = vec(b), vec(p)
    return out


def test_holm():
    adj = rq1.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert rq1.holm({"x": 0.6, "y": 0.5}) == pytest.approx({"y": 1.0, "x": 1.0})


def test_mixed_case():
    res = rq1.analyse(cells(), cross="D3")
    assert res["delta"]["D0"]["estimate"] == pytest.approx(0.40)
    assert res["H1"]["supported"]                                   # 40 pp >= 30 pp
    assert res["H2"]["supported"] and res["H2"]["p"] == 0.0         # 40 - 20 pp, never <= 0
    assert res["H3"]["supported"]                                   # 1 pp, far below 5 pp
    assert not res["H4"]["supported"] and res["H4"]["p"] == 1.0     # 40 pp
    assert res["conclusion"] == "mixed" and res["conclusion_text"] is None
    assert res["cross"]["estimate"] == pytest.approx(240 / N - 230 / N)
    lo, hi = res["delta"]["D3"]["ci95"]
    assert lo <= 0.01 <= hi


def test_termination_and_edit_cases():
    assert rq1.analyse(cells(d2b=(300, 300)))["conclusion"] == "termination"
    res = rq1.analyse(cells(d3=(100, 300)))
    assert res["conclusion"] == "edit" and res["conclusion_text"].startswith("The edit retains")


def test_margin_is_one_sided_on_the_upper_side():
    # an effect of exactly 5 pp is not "below 5 pp"
    res = rq1.analyse(cells(d3=(200, 225)))
    assert res["H3"]["p"] > 0.4 and not res["H3"]["supported"]


def test_deterministic():
    a, b = rq1.analyse(cells()), rq1.analyse(cells())
    assert a == b


def test_problem_accuracy_needs_every_sample():
    recs = [{"problem_id": "p1", "correct": True}, {"problem_id": "p1", "correct": False},
            {"problem_id": "p2", "correct": True}]
    with pytest.raises(ValueError):
        rq1.problem_accuracy(recs, ["p1", "p2"], samples=2)
    assert list(rq1.problem_accuracy(recs, ["p1", "p2"], samples=1)) == [0.5, 1.0]


def test_choices():
    assert rq1.choose_tau({0.3: 50, 0.5: 60, 0.7: 60}) == 0.7          # tie -> larger tau
    assert rq1.choose_tau({0.3: 70, 0.5: 60, 0.7: 60}) == 0.3
    assert rq1.choose_config({"D1": 40, "D2b": 50, "D3": 50}) == "D2b"  # tie -> order D1, D2b, D3
    assert rq1.choose_config({"D1": 50, "D2b": 50, "D3": 50}) == "D1"


def test_both_finished():
    def r(model, pid, correct, cut):
        return {"model": model, "config": "D1", "benchmark": "math500", "problem_id": pid, "correct": correct,
                "cut": cut}
    groups = {("base", "D1", "math500"): [r("base", "a", True, False), r("base", "b", False, True)],
              ("p50c", "D1", "math500"): [r("p50c", "a", False, False), r("p50c", "b", True, False)]}
    assert rq1.both_finished(groups) == {"D1/math500": {"problems": 1, "base": 1.0, "p50c": 0.0}}
