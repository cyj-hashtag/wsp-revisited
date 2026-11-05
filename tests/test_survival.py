"""RQ3 survival analysis and decomposition (src/wsprev/survival.py) on hand-computable and synthetic data."""
import json

import numpy as np
import pytest

from wsprev import survival as sv

# 6-MP group of Freireich et al. (1963), the classic Kaplan-Meier example (Gehan 1965): weeks of remission,
# E = 0 marks a censored time. Every expected value below is computed by hand from these 21 times.
SIX_MP_T = [6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35]
SIX_MP_E = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_km_textbook_example():
    km = sv.kaplan_meier(SIX_MP_T, SIX_MP_E)
    assert km.t.tolist() == [6, 7, 10, 13, 16, 22, 23]
    assert km.n.tolist() == [21, 17, 15, 12, 11, 7, 6]
    assert km.d.tolist() == [3, 1, 1, 1, 1, 1, 1]
    s, expect = 1.0, []
    for n, d in zip([21, 17, 15, 12, 11, 7, 6], [3, 1, 1, 1, 1, 1, 1]):
        s *= 1 - d / n
        expect.append(s)
    assert km.s == pytest.approx(expect)
    assert km.s[0] == pytest.approx(18 / 21)
    # Greenwood at t = 6: S^2 * 3 / (21 * 18)
    assert km.var([6])[0] == pytest.approx((18 / 21) ** 2 * 3 / (21 * 18))
    g23 = 3 / (21 * 18) + 1 / (17 * 16) + 1 / (15 * 14) + 1 / (12 * 11) + 1 / (11 * 10) + 1 / (7 * 6) + 1 / (6 * 5)
    assert km.var([23, 40])[0] == pytest.approx(expect[-1] ** 2 * g23)
    # step function: S = 1 before 6, constant between event times
    assert km([0, 5.9, 6, 8, 23, 100]).tolist() == pytest.approx([1, 1, 18 / 21, expect[1], expect[-1], expect[-1]])
    assert sv.median(km) == 23          # S(22) = 0.538 > 0.5 >= S(23) = 0.448
    assert km.n_events == 9 and km.t_max == 35


def test_km_band_is_loglog_and_contains_estimate():
    km = sv.kaplan_meier(SIX_MP_T, SIX_MP_E)
    lo, hi = km.band(km.t)
    assert np.all(lo <= km.s) and np.all(km.s <= hi) and np.all(lo >= 0) and np.all(hi <= 1)
    # symmetric in log(-log S): theta +- z sigma / |log S|
    th = np.log(-np.log(km.s))
    w = sv.Z95 * np.sqrt(km.gw) / np.abs(np.log(km.s))
    assert np.log(-np.log(lo)) == pytest.approx(th + w)
    assert np.log(-np.log(hi)) == pytest.approx(th - w)
    lo0, hi0 = km.band([0])
    assert lo0[0] == hi0[0] == 1.0      # no event yet


def test_km_without_censoring_is_the_empirical_survival():
    rng = np.random.default_rng(1)
    T = rng.integers(1, 50, size=300)
    km = sv.kaplan_meier(T, np.ones_like(T))
    x = np.arange(0, 60)
    assert km(x) == pytest.approx([(T > v).mean() for v in x])
    assert km.s[-1] == 0 and km.var([100])[0] == 0      # S reaches 0: variance 0, no NaN


def test_km_all_censored():
    km = sv.kaplan_meier([5, 7, 9], [0, 0, 0])
    assert km(np.array([0, 10])).tolist() == [1.0, 1.0] and sv.median(km) is None
    assert sv.rmst(km, 20) == 20


def test_nelson_aalen_textbook_example():
    t, H, V = sv.nelson_aalen(SIX_MP_T, SIX_MP_E)
    n = np.array([21, 17, 15, 12, 11, 7, 6])
    d = np.array([3, 1, 1, 1, 1, 1, 1])
    assert H == pytest.approx(np.cumsum(d / n))
    assert V == pytest.approx(np.cumsum(d / n ** 2))
    assert sv.na_at(SIX_MP_T, SIX_MP_E, [0, 6, 12]).tolist() == pytest.approx([0, 3 / 21, 3 / 21 + 1 / 17 + 1 / 15])


def test_smoothed_hazard_recovers_a_constant_hazard():
    rng = np.random.default_rng(2)
    lam = 1 / 4000
    T = np.ceil(rng.exponential(1 / lam, size=200_000))
    E = T <= 20000
    T = np.minimum(T, 20000)
    h = sv.smoothed_hazard(T, E, [5000, 10000], bandwidth=1024)
    assert h == pytest.approx([lam, lam], rel=0.05)
    assert np.isnan(sv.smoothed_hazard(T, E, [30000])[0])     # beyond the last observed time


def test_rmst_with_administrative_censoring_is_mean_of_min():
    rng = np.random.default_rng(3)
    T = rng.integers(1, 40000, size=1000)
    L = 32768
    km = sv.kaplan_meier(np.minimum(T, L), T < L)
    assert sv.rmst(km, L) == pytest.approx(np.minimum(T, L).mean())


def rec(think_closed=True, finish="stop", think=100, n_gen=150, trig=None, cut=None, correct=True, pid="p0", s=0,
        model="base", config="D0", bench="math500"):
    return {"model": model, "config": config, "benchmark": bench, "problem_id": pid, "sample_idx": s,
            "think_closed": think_closed, "finish_reason": finish, "think_tokens": think, "n_gen_tokens": n_gen,
            "loop_triggered": trig is not None, "trigger_pos": trig, "correct": correct,
            "cut": (finish == "length") if cut is None else cut}


def test_durations_cover_every_ending():
    recs = [rec(),                                                      # </think> at 100
            rec(think_closed=True, finish="length", think=32000, n_gen=32768),   # closed, answer cut
            rec(think_closed=False, finish="stop", think=900, n_gen=900),        # end of text inside thinking
            rec(think_closed=False, finish="length", think=32768, n_gen=32768),  # never closed
            rec(think_closed=False, finish="stop", think=4096, n_gen=4200, trig=4096),   # D3 breaker
            rec(think_closed=None, finish="stop", think=0, n_gen=700),           # thinking off, finished
            rec(think_closed=None, finish="length", think=0, n_gen=32768)]       # thinking off, cut
    T, E = sv.durations(recs)
    assert T.tolist() == [100, 32000, 900, 32768, 4096, 700, 32768]
    assert E.tolist() == [True, True, True, False, False, True, False]
    assert sv.durations(recs, "event")[1].tolist() == [True, True, True, False, True, True, False]
    assert sv.ending_counts(recs) == {"</think>": 2, "end of text without </think>": 1, "cut": 2, "breaker": 1,
                                      "end of text": 1}
    with pytest.raises(ValueError):
        sv.durations(recs, "drop")


def arr(*v):
    return np.array(v, dtype=float)


def test_shapley_matches_the_generic_formula_and_adds_up():
    rng = np.random.default_rng(4)
    for _ in range(20):
        p0, p1, aF0, aF1, aC0, aC1 = rng.random(6)
        f0 = {"p": arr(p0), "aF": arr(aF0), "aC": arr(aC0), "acc": arr(p0 * aF0 + (1 - p0) * aC0)}
        f1 = {"p": arr(p1), "aF": arr(aF1), "aC": arr(aC1), "acc": arr(p1 * aF1 + (1 - p1) * aC1)}
        s = sv.shapley(f0, f1)
        phi_p, phi_a = sv.shapley_generic(lambda p, a: p * a[0] + (1 - p) * a[1], p0, (aF0, aC0), p1, (aF1, aC1))
        assert s["termination"][0] == pytest.approx(phi_p)
        assert s["conditional"][0] == pytest.approx(phi_a)
        assert s["termination"][0] + s["conditional"][0] == pytest.approx(s["delta"][0])
        assert s["conditional_finished"][0] + s["conditional_cut"][0] == pytest.approx(s["conditional"][0])
        back = sv.shapley(f1, f0)          # antisymmetric
        assert back["termination"][0] == pytest.approx(-s["termination"][0])


def test_shapley_pure_termination():
    # same conditional accuracies, only p changes: everything is termination
    f0 = {"p": arr(0.1), "aF": arr(0.9), "aC": arr(0.0), "acc": arr(0.09)}
    f1 = {"p": arr(0.7), "aF": arr(0.9), "aC": arr(0.0), "acc": arr(0.63)}
    s = sv.shapley(f0, f1)
    assert s["termination"][0] == pytest.approx(0.54) and s["conditional"][0] == pytest.approx(0)


def two_models(n_problems=40, samples=2, config="D1", seed=5):
    """Synthetic base / p50c records: base finishes 20% of samples, p50c 70%; finished answers are right with
    probability 0.9, cut ones never."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_problems):
        for s in range(samples):
            for m, pf in (("base", 0.2), ("p50c", 0.7)):
                fin = rng.random() < pf
                L = int(rng.integers(500, 30000)) if fin else 32768
                recs.append(rec(think_closed=fin, finish="stop" if fin else "length", think=L, n_gen=L,
                                correct=fin and rng.random() < 0.9, pid=f"p{i:03d}", s=s, model=m, config=config))
    return recs


def test_problem_stats_factors_and_cells_add_up():
    recs = two_models()
    ids = sorted({r["problem_id"] for r in recs})
    b = [r for r in recs if r["model"] == "base"]
    p = [r for r in recs if r["model"] == "p50c"]
    f = {m: sv.factors(sv.problem_stats(x, ids)) for m, x in (("base", b), ("p50c", p))}
    for m, x in (("base", b), ("p50c", p)):
        acc = np.mean([r["correct"] for r in x])
        assert f[m]["acc"][0] == pytest.approx(acc)
        assert f[m]["p"][0] * f[m]["aF"][0] + (1 - f[m]["p"][0]) * f[m]["aC"][0] == pytest.approx(acc)
    cells = sv.cells_of(sv.pair_stats(b, p, ids))
    delta = f["p50c"]["acc"][0] - f["base"]["acc"][0]
    assert sum(cells[c]["contribution"][0] for c in sv.CELLS) == pytest.approx(delta)
    assert sum(cells[c]["share"][0] for c in sv.CELLS) == pytest.approx(1)
    one = cells[sv.ONE_FINISHED]
    assert one["contribution"][0] == pytest.approx(cells[sv.CELLS[1]]["contribution"][0]
                                                   + cells[sv.CELLS[2]]["contribution"][0])
    assert one["share"][0] == pytest.approx(cells[sv.CELLS[1]]["share"][0] + cells[sv.CELLS[2]]["share"][0])


def test_budget_counts_correct_and_finished_within_L():
    recs = [rec(n_gen=1000, think=900), rec(n_gen=5000, think=4000, s=1), rec(n_gen=20000, think=19000, s=2),
            rec(n_gen=32768, think=32768, think_closed=False, finish="length", correct=True, s=3)]  # cut, graded correct
    st = sv.problem_stats(recs, ["p0"], budgets=[0, 4096, 8192, 32768])
    assert st["within"][0].tolist() == [0, 1, 2, 3]
    assert st["cc"][0] == 1 and st["cf"][0] == 3


def test_boot_weights_reproduce_the_registered_draw():
    w = sv.boot_weights(7, n_boot=50)
    idx = np.random.default_rng(0).integers(0, 7, size=(50, 7))
    assert w.sum(axis=1).tolist() == [7] * 50
    assert w[3].tolist() == np.bincount(idx[3], minlength=7).tolist()


def test_ci_drops_undefined_replicates():
    c = sv.ci(np.array([np.nan, 0.1, 0.2, 0.3]))
    assert c["undefined_replicates"] == 1 and c["ci95"][0] >= 0.1


def test_pair_stats_needs_matching_samples():
    with pytest.raises(ValueError, match="samples"):
        sv.pair_stats([rec(s=0), rec(s=1)], [rec(s=0, model="p50c")], ["p0"])


def run_main(tmp_path):
    """wsprev.survival.main on synthetic D0, D1 and D3 records; returns the output folder."""
    runs = tmp_path / "runs"
    d0 = [r for r in two_models(samples=1, config="D0", seed=6)]
    rq1 = two_models(config="D1", seed=7) + two_models(config="D3", seed=8)
    for r in rq1:     # turn some D3 cut samples into breaker cuts
        if r["config"] == "D3" and r["cut"] and r["problem_id"] < "p010":
            r.update(loop_triggered=True, trigger_pos=3072, think_tokens=3072, think_closed=False)
    for folder, rs in (("d0", d0), ("rq1", rq1)):
        (runs / folder / "x").mkdir(parents=True)
        (runs / folder / "x" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    out, summary = tmp_path / "rq3", tmp_path / "rq3_summary.md"
    sv.main(["--runs", str(runs), "--out", str(out), "--summary", str(summary), "--n-boot", "200"])
    res = json.loads((out / "rq3.json").read_text())
    for key in ("D0/math500", "D1/math500", "D3/math500"):
        d = res["decomposition"][key]
        assert d["termination"]["estimate"] + d["conditional"]["estimate"] == pytest.approx(d["delta"]["estimate"])
        assert sum(res["cells"][key][c]["contribution"]["estimate"] for c in sv.CELLS) == \
            pytest.approx(d["delta"]["estimate"])
        lo, hi = d["termination"]["ci95"]
        assert lo <= d["termination"]["estimate"] <= hi
    assert "base/D3/math500/breaker-as-end" in res["km"]
    assert res["d0_d3_identity"]["base"]["compared"] > 0
    text = summary.read_text(encoding="utf-8")
    assert "Shapley decomposition" in text and "Budget curve" in text
    assert (out / "km_curves.csv").read_text().startswith("model,config,benchmark,ending,t,at_risk,S")
    assert len((out / "budget_curves.csv").read_text().splitlines()) == 1 + 3 * 2 * len(sv.GRID)
    return out


def test_main_end_to_end(tmp_path):
    run_main(tmp_path)


def test_plot_rq3_draws_every_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from test_scripts import script

    out = run_main(tmp_path)
    # the figures need D2b on MATH-500 as well: reuse D1's rows under that name
    for name in ("budget_curves.csv", "km_curves.csv"):
        lines = (out / name).read_text().splitlines()
        lines += [ln.replace(",D1,", ",D2b,") for ln in lines[1:] if ",D1,math500," in ln]
        (out / name).write_text("\n".join(lines) + "\n")
    res = json.loads((out / "rq3.json").read_text())
    for part in ("decomposition", "cells"):
        res[part]["D2b/math500"] = res[part]["D1/math500"]
    (out / "rq3.json").write_text(json.dumps(res))
    script("plot_rq3").main(["--rq3", str(out)])
    for name in ("survival", "survival_d3", "hazard", "decomposition", "budget"):
        assert (out / "figs" / f"{name}.pdf").stat().st_size > 1000
        assert (out / "figs" / f"{name}.png").stat().st_size > 1000
