"""The registered RQ1 analysis (prereg/criteria.md section 5) on synthetic per-problem accuracies."""
import json

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


def rec(model, config, bench, pid, k, correct):
    return {"model": model, "config": config, "benchmark": bench, "problem_id": pid, "sample_idx": k,
            "correct": correct, "cut": False}


def test_effect_ci_uses_the_registered_resampling():
    c = cells()
    res = rq1.analyse(c)
    for cfg in rq1.H_CONFIGS:
        e = rq1.effect_ci(c[("base", cfg)], c[("p50c", cfg)])
        assert e["estimate"] == pytest.approx(res["delta"][cfg]["estimate"])
        assert e["ci95"] == pytest.approx(res["delta"][cfg]["ci95"])


def test_secondary_effects(tmp_path):
    ids = ["gsm8k_test_0", "gsm8k_test_1", "gsm8k_test_2", "gsm8k_test_3"]
    groups = {}
    for m, solved in (("base", 1), ("p50c", 3)):                       # GSM8K D1: one sample per problem
        groups[(m, "D1", "gsm8k")] = [rec(m, "D1", "gsm8k", i, 0, n < solved) for n, i in enumerate(ids)]
    groups[("base", "D2b", "gsm8k")] = groups[("base", "D1", "gsm8k")]  # p50c missing -> incomplete
    for folder, correct in (("unedited", [0, 0]), ("p50c", [1, 0])):   # the original evaluation: 2 samples
        part = tmp_path / folder / "parts" / "gsm8k_s0of1"
        part.mkdir(parents=True)
        part.joinpath("gsm8k_per_problem.json").write_text(json.dumps(
            [{"id": i, "correct": correct, "pred_first": None} for i in ids]))
    sec = rq1.secondary_effects(groups, {"gsm8k": ids}, tmp_path)
    assert sec["D1/gsm8k"]["estimate"] == pytest.approx(0.5) and sec["D1/gsm8k"]["source"] == "this study"
    assert sec["D0/gsm8k"]["estimate"] == pytest.approx(0.5) and sec["D0/gsm8k"]["source"] == "original evaluation"
    assert sec["D0/gsm8k"]["base"] == 0.0 and sec["D0/gsm8k"]["p50c"] == 0.5
    assert "p50c" in sec["D2b/gsm8k"]["incomplete"]
    assert "D3/gsm8k" not in sec and "D2a/gsm8k" not in sec
    short = {("base", "D1", "gsm8k"): groups[("base", "D1", "gsm8k")][:2], ("p50c", "D1", "gsm8k"): []}
    assert "fewer than 1 samples" in rq1.secondary_effects(short, {"gsm8k": ids})["D1/gsm8k"]["incomplete"]


def test_markdown_uses_latex():
    res = rq1.analyse(cells(), cross="D3")
    res.update(tau=0.5, git="abc1234")
    sec = {"D1/gsm8k": {"problems": 4, "base": 0.25, "p50c": 0.75, "estimate": 0.5, "ci95": [0.0, 1.0],
                        "source": "this study"}, "D2b/gsm8k": {"incomplete": "no runs of ['p50c']"}}
    md = rq1.markdown(res, {}, {}, sec)
    assert r"| H3 | $\Delta_{\mathrm{D3}} \lt m$ |" in md
    assert r"$\operatorname{Acc}(\mathrm{p50c}, \mathrm{D0}) - \operatorname{Acc}(\mathrm{base}, \mathrm{D3}) = " in md
    assert "| gsm8k | D1 | 4 |" in md and "incomplete: no runs of ['p50c']" in md
    assert "Δ_" not in md and "≥" not in md
