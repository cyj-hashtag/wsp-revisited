"""RQ3: survival analysis of the thinking length, and where the edit's accuracy gain comes from
(docs/plan-for-agents.md, section 3, RQ3).

    python -m wsprev.survival [--runs runs] [--out results/rq3]

Reads every records.jsonl under <runs>/d0 (D0, MATH-500, 1 sample) and <runs>/rq1 (D1, D2a, D2b, D3), that is the
records archive of the GPU run unpacked in the repository (`tar xzf runs/gpushare_2026-11-04/runs_records.tgz`), and
writes results/rq3/rq3.json, km_curves.csv, budget_curves.csv and results/rq3_summary.md. The figures are drawn from
these files by scripts/plot_rq3.py. Benchmark files are not needed.

Definitions (one record = one sample; the record fields are listed in plan section 5). They are choices for the author
to check; each is a parameter or a separate function.

- Thinking length T and event E. Thinking on (D0, D1, D3): T = think_tokens, the tokens before `</think>`. E = 1 if
  the model ended its thinking by itself: it wrote `</think>`, or it stopped (end of text) without one. Otherwise T is
  right-censored: at the token limit L = 32768 (D0, D1), or at the breaker's cut (D3, breaker="censor"). With
  breaker="event" the breaker's cut counts as an end of thinking (D3's thinking length as generated). Thinking off
  (D2a, D2b): T = n_gen_tokens, E = 1 if it ended with end of text.
- Kaplan-Meier: S(t) = prod_{t_i <= t} (1 - d_i / n_i) over the distinct event times t_i, with d_i events at t_i and
  n_i = #{T >= t_i} at risk (a sample censored at t_i is still at risk at t_i). Greenwood:
  Var S(t) = S(t)^2 sum_{t_i <= t} d_i / (n_i (n_i - d_i)). The 95% band is the log-log one, symmetric in
  log(-log S). Samples of one problem are treated as independent here (MATH-500 has 2 per problem outside D0), so the
  band is slightly too narrow; the bootstrap intervals below resample problems instead.
- Nelson-Aalen: H(t) = sum_{t_i <= t} d_i / n_i, Var H(t) = sum d_i / n_i^2. The hazard h(t) is the kernel smoothing
  of the increments (Ramlau-Hansen): h(t) = (1/b) sum_i K((t - t_i) / b) d_i / n_i with the Epanechnikov kernel
  K(x) = 3/4 (1 - x^2) on |x| <= 1, bandwidth b = 1024 tokens, no boundary correction (biased within b of 0 and L).
- Finished F = 1 if the answer was not cut at the token limit (the record's `cut` is false): the "Acc. finished" of
  results/rq1_summary.md. Then Acc = p a_F + (1 - p) a_C with p = P(F), a_F = P(correct | F), a_C = P(correct | not F).
- Shapley decomposition of Delta = Acc(p50c) - Acc(base) over the two factors p and a = (a_F, a_C), with
  f(p, a) = p a_F + (1 - p) a_C:
      termination  = (p1 - p0) [(a_F0 - a_C0) + (a_F1 - a_C1)] / 2
      conditional  = pbar (a_F1 - a_F0) + (1 - pbar) (a_C1 - a_C0),  pbar = (p0 + p1) / 2,
  which add up to Delta. Undefined (None) when a group has no finished or no cut sample.
- Paired cells: sample s of base and sample s of p50c on the same problem form a pair (1 000 pairs on MATH-500
  outside D0). With (F0, F1) the pair's finishing status, Delta = sum over the 4 cells of
  P(cell) [P(correct_1 | cell) - P(correct_0 | cell)] exactly. The cell "both finished" compares the two models on the
  same pairs, free of the selection in a_F (the base model finishes mostly easy problems).
- Problems where neither model was cut (prereg section 5's "both finish"), with the paired difference.
- Budget curve: Acc(L) = P(correct, finished, n_gen_tokens <= L), the share of samples that end with a correct answer
  within L generated tokens. For D0, D1, D2a and D2b this is the accuracy a run with the token limit L would have had,
  except for answers cut at L and still graded correct (the grader reads the whole text when `</think>` is missing);
  at L = 32768 it is Acc - P(correct, cut). For D3 it is a description only (a run with limit L would also change
  where the breaker cuts).

Bootstrap: the problems of a benchmark in sorted order, one draw of numpy.random.default_rng(0) of shape
(10 000, N) shared by every model, configuration and statistic of that benchmark (the scheme of prereg section 5; the
problem order differs from rq1.py's, which uses the benchmark file's, so intervals of Delta itself can differ from
results/rq1_summary.md in the last digit). Ratios use the resampled sums; a replicate with an empty denominator is
dropped and counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._vendor import REPO
from .summary import group, read_records

L_MAX = 32768
N_BOOT, BOOT_SEED = 10_000, 0
GRID = np.arange(0, L_MAX + 1, 256)                 # time / budget grid of the curves (tokens)
S_POINTS = (2048, 4096, 8192, 16384, 30720)         # S(t) reported in the tables
BUDGETS = (4096, 8192, 16384, 32768)                # Acc(L) reported in the tables (plan section 3)
BANDWIDTH = 1024
Z95 = 1.959963984540054
MODELS = ("base", "p50c")
CONFIGS = ("D0", "D1", "D2a", "D2b", "D3")
BENCHES = ("math500", "gsm8k", "ifeval")
CELLS = ("both finished", "only p50c finished", "only base finished", "both cut")
ONE_FINISHED = "one finished"      # the union of the two middle cells


# ---------------------------------------------------------------- estimators
@dataclass
class KM:
    """Kaplan-Meier estimate. Arrays are over the distinct event times t (ascending)."""
    t: np.ndarray
    n: np.ndarray        # at risk just before t
    d: np.ndarray        # events at t
    s: np.ndarray        # S(t), right-continuous
    gw: np.ndarray       # Greenwood sum: sum_{t_i <= t} d_i / (n_i (n_i - d_i)); inf once S reaches 0
    n_total: int
    n_events: int
    t_max: float         # largest observed time (event or censoring)

    def __call__(self, x):
        """S at the times x (S = 1 before the first event)."""
        k = np.searchsorted(self.t, np.asarray(x, dtype=float), side="right")
        return np.concatenate([[1.0], self.s])[k]

    def var(self, x):
        """Greenwood variance of S at the times x (0 where S = 0 or no event yet)."""
        k = np.searchsorted(self.t, np.asarray(x, dtype=float), side="right")
        s = np.concatenate([[1.0], self.s])[k]
        g = np.concatenate([[0.0], self.gw])[k]
        with np.errstate(invalid="ignore"):
            return np.where(s > 0, s ** 2 * g, 0.0)

    def band(self, x, z: float = Z95):
        """Pointwise log-log band (lower, upper): S^exp(+-z sigma / |log S|), sigma^2 the Greenwood sum."""
        k = np.searchsorted(self.t, np.asarray(x, dtype=float), side="right")
        s = np.concatenate([[1.0], self.s])[k]
        g = np.concatenate([[0.0], self.gw])[k]
        lo, hi = s.copy(), s.copy()
        ok = (s > 0) & (s < 1) & np.isfinite(g)
        w = z * np.sqrt(g[ok]) / np.abs(np.log(s[ok]))
        lo[ok], hi[ok] = s[ok] ** np.exp(w), s[ok] ** np.exp(-w)
        return lo, hi


def at_risk(T, x) -> np.ndarray:
    """#{T >= x} for every x: the risk set just before x."""
    T = np.sort(np.asarray(T, dtype=float))
    return len(T) - np.searchsorted(T, np.asarray(x, dtype=float), side="left")


def kaplan_meier(T, E) -> KM:
    T, E = np.asarray(T, dtype=float), np.asarray(E, dtype=bool)
    if T.shape != E.shape or T.ndim != 1 or len(T) == 0:
        raise ValueError("T and E must be non-empty 1-d arrays of the same length")
    t = np.unique(T[E])
    n = at_risk(T, t)
    d = np.bincount(np.searchsorted(t, T[E]), minlength=len(t)).astype(float)
    s = np.cumprod(1.0 - d / n)
    with np.errstate(divide="ignore"):
        gw = np.cumsum(np.where(n > d, d / (n * (n - d)), np.inf))
    return KM(t, n.astype(float), d, s, gw, len(T), int(E.sum()), float(T.max()))


def nelson_aalen(T, E) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(event times, H at them, Var H at them)."""
    km = kaplan_meier(T, E)
    return km.t, np.cumsum(km.d / km.n), np.cumsum(km.d / km.n ** 2)


def na_at(T, E, x) -> np.ndarray:
    t, H, _ = nelson_aalen(T, E)
    k = np.searchsorted(t, np.asarray(x, dtype=float), side="right")
    return np.concatenate([[0.0], H])[k]


def smoothed_hazard(T, E, x, bandwidth: float = BANDWIDTH) -> np.ndarray:
    """Kernel-smoothed Nelson-Aalen hazard (per token) at the times x; NaN beyond the largest observed time."""
    km = kaplan_meier(T, E)
    x = np.asarray(x, dtype=float)
    u = (x[:, None] - km.t[None, :]) / bandwidth
    k = np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
    h = (k * (km.d / km.n)[None, :]).sum(axis=1) / bandwidth
    return np.where(x <= km.t_max, h, np.nan)


def rmst(km: KM, tau: float = L_MAX) -> float:
    """Restricted mean of T: the integral of S from 0 to tau (with censoring only at tau, the mean of min(T, tau))."""
    edges = np.concatenate([[0.0], km.t[km.t < tau], [tau]])
    heights = np.concatenate([[1.0], km.s[km.t < tau]])
    return float(np.sum(heights * np.diff(edges)))


def median(km: KM) -> float | None:
    k = np.nonzero(km.s <= 0.5)[0]
    return float(km.t[k[0]]) if len(k) else None


# ---------------------------------------------------------------- records -> durations
def durations(recs, breaker: str = "censor") -> tuple[np.ndarray, np.ndarray]:
    """(T, E) of the records (module docstring). breaker: "censor" or "event" (D3's cut)."""
    if breaker not in ("censor", "event"):
        raise ValueError(breaker)
    T, E = [], []
    for r in recs:
        if r["think_closed"] is None:                       # thinking off
            T.append(r["n_gen_tokens"])
            E.append(r["finish_reason"] == "stop")
        elif r.get("loop_triggered"):                       # D3: the breaker cut the thinking
            T.append(r["trigger_pos"])
            E.append(breaker == "event")
        else:
            T.append(r["think_tokens"])
            E.append(bool(r["think_closed"]) or r["finish_reason"] == "stop")
    return np.array(T, dtype=float), np.array(E, dtype=bool)


def ending_counts(recs) -> dict:
    """How the thinking ended: by `</think>`, by end of text without `</think>`, by the breaker, or not (cut)."""
    c = defaultdict(int)
    for r in recs:
        if r["think_closed"] is None:
            c["end of text" if r["finish_reason"] == "stop" else "cut"] += 1
        elif r.get("loop_triggered"):
            c["breaker"] += 1
        elif r["think_closed"]:
            c["</think>"] += 1
        elif r["finish_reason"] == "stop":
            c["end of text without </think>"] += 1
        else:
            c["cut"] += 1
    return dict(c)


def km_summary(km: KM, T) -> dict:
    lo, hi = km.band(S_POINTS)
    return {"samples": km.n_total, "events": km.n_events, "censored": km.n_total - km.n_events,
            "S": {str(t): float(v) for t, v in zip(S_POINTS, km(S_POINTS))},
            "S_ci95": {str(t): [float(a), float(b)] for t, a, b in zip(S_POINTS, lo, hi)},
            "S_end": float(km.s[-1]) if len(km.s) else 1.0,
            "median": median(km), "rmst": rmst(km), "mean_T": float(np.mean(T))}


# ---------------------------------------------------------------- per-problem statistics and the bootstrap
def by_problem(recs) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in recs:
        out[r["problem_id"]].append(r)
    return out


def problem_stats(recs, ids, budgets=GRID) -> dict[str, np.ndarray]:
    """Per problem (rows in the order of ids): samples n, finished f, correct and finished cf, correct and cut cc,
    and correct-finished-within-L counts for every L of `budgets` (columns)."""
    per = by_problem(recs)
    missing = [i for i in ids if i not in per]
    if missing:
        raise ValueError(f"{len(missing)} problems have no sample, e.g. {missing[:3]}")
    budgets = np.asarray(budgets)
    st = {k: np.zeros(len(ids)) for k in ("n", "f", "cf", "cc")}
    st["within"] = np.zeros((len(ids), len(budgets)))
    for j, i in enumerate(ids):
        for r in per[i]:
            fin, ok = not r["cut"], bool(r["correct"])
            st["n"][j] += 1
            st["f"][j] += fin
            st["cf"][j] += fin and ok
            st["cc"][j] += (not fin) and ok
            if fin and ok:
                st["within"][j] += r["n_gen_tokens"] <= budgets
    return st


def boot_weights(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Multiplicity of each problem in each replicate (n_boot x n) of one draw of default_rng(seed) of shape
    (n_boot, n): the resampling of prereg section 5."""
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    w = np.zeros((n_boot, n))
    for b in range(n_boot):
        w[b] = np.bincount(idx[b], minlength=n)
    return w


def ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def ci(x) -> dict:
    x = np.asarray(x, dtype=float)
    ok = x[np.isfinite(x)]
    out = {"ci95": [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else None}
    if len(ok) < len(x):
        out["undefined_replicates"] = int(len(x) - len(ok))
    return out


def factors(st, w=None) -> dict[str, np.ndarray]:
    """p, a_F, a_C and Acc from the per-problem sums (w: bootstrap weights, else the sample itself)."""
    s = {k: (st[k].sum() if w is None else w @ st[k]) for k in ("n", "f", "cf", "cc")}
    s = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in s.items()}
    return {"p": s["f"] / s["n"], "aF": ratio(s["cf"], s["f"]), "aC": ratio(s["cc"], s["n"] - s["f"]),
            "acc": (s["cf"] + s["cc"]) / s["n"]}


def shapley(f0: dict, f1: dict) -> dict[str, np.ndarray]:
    """Two-factor Shapley decomposition of Acc1 - Acc0 into termination (p) and conditional accuracy (a_F, a_C)."""
    p0, p1 = f0["p"], f1["p"]
    pbar = (p0 + p1) / 2
    term = (p1 - p0) * ((f0["aF"] - f0["aC"]) + (f1["aF"] - f1["aC"])) / 2
    cond_f = pbar * (f1["aF"] - f0["aF"])
    cond_c = (1 - pbar) * (f1["aC"] - f0["aC"])
    return {"delta": f1["acc"] - f0["acc"], "termination": term, "conditional": cond_f + cond_c,
            "conditional_finished": cond_f, "conditional_cut": cond_c}


def shapley_generic(f, a0, b0, a1, b1) -> tuple[float, float]:
    """Shapley values of two factors for any f(a, b) (used by the tests to check shapley)."""
    phi_a = ((f(a1, b0) - f(a0, b0)) + (f(a1, b1) - f(a0, b1))) / 2
    phi_b = ((f(a0, b1) - f(a0, b0)) + (f(a1, b1) - f(a1, b0))) / 2
    return phi_a, phi_b


def pair_stats(base_recs, p50c_recs, ids) -> dict[str, np.ndarray]:
    """Per problem: pairs, and per cell (CELLS) the pair count and the correct counts of base (y0) and p50c (y1).
    Pairs are (sample s of base, sample s of p50c); a problem with unequal sample counts is an error."""
    pb, pp = by_problem(base_recs), by_problem(p50c_recs)
    st = {"pairs": np.zeros(len(ids))}
    for c in CELLS:
        st[c] = np.zeros((len(ids), 3))   # count, y0, y1
    for j, i in enumerate(ids):
        b = {r["sample_idx"]: r for r in pb[i]}
        p = {r["sample_idx"]: r for r in pp[i]}
        if set(b) != set(p):
            raise ValueError(f"{i}: base samples {sorted(b)} vs p50c samples {sorted(p)}")
        for s in b:
            f0, f1 = not b[s]["cut"], not p[s]["cut"]
            cell = CELLS[0] if f0 and f1 else CELLS[1] if f1 else CELLS[2] if f0 else CELLS[3]
            st["pairs"][j] += 1
            st[cell][j] += (1, bool(b[s]["correct"]), bool(p[s]["correct"]))
    return st


def cells_of(st, w=None) -> dict[str, dict[str, np.ndarray]]:
    """Per cell: share of pairs, accuracy of base and p50c in the cell, and its contribution to Delta; also the union
    "one finished" of the two cells where exactly one model finished."""
    n = np.atleast_1d(st["pairs"].sum() if w is None else w @ st["pairs"])
    st = dict(st, **{ONE_FINISHED: st[CELLS[1]] + st[CELLS[2]]})
    out = {}
    for c in CELLS + (ONE_FINISHED,):
        s = st[c].sum(axis=0)[None, :] if w is None else w @ st[c]
        out[c] = {"share": s[:, 0] / n, "acc_base": ratio(s[:, 1], s[:, 0]), "acc_p50c": ratio(s[:, 2], s[:, 0]),
                  "contribution": (s[:, 2] - s[:, 1]) / n}
    return out


def neither_cut(base_recs, p50c_recs, ids) -> list[str]:
    """Problems where no sample of either model was cut (prereg section 5)."""
    pb, pp = by_problem(base_recs), by_problem(p50c_recs)
    return [i for i in ids if not any(r["cut"] for r in pb[i] + pp[i])]


# ---------------------------------------------------------------- analysis
def scalar(x):
    x = np.atleast_1d(x)
    return None if not np.isfinite(x[0]) else float(x[0])


def analyse(recs, n_boot: int = N_BOOT) -> dict:
    groups = group(recs)
    out = {"km": {}, "endings": {}, "decomposition": {}, "cells": {}, "neither_cut": {}, "budget": {},
           "curves": [], "budget_curves": []}
    for bench in BENCHES:
        cfgs = [c for c in CONFIGS if all((m, c, bench) in groups for m in MODELS)]
        if not cfgs:
            continue
        ids = sorted({r["problem_id"] for c in cfgs for m in MODELS for r in groups[(m, c, bench)]})
        w = boot_weights(len(ids), n_boot)
        for c in cfgs:
            key = f"{c}/{bench}"
            rs = {m: groups[(m, c, bench)] for m in MODELS}
            st = {m: problem_stats(rs[m], ids) for m in MODELS}
            # survival
            for m in MODELS:
                for mode in (("censor", "event") if c == "D3" else ("censor",)):
                    T, E = durations(rs[m], mode)
                    km = kaplan_meier(T, E)
                    name = f"{m}/{c}/{bench}" + ("" if mode == "censor" else "/breaker-as-end")
                    out["km"][name] = km_summary(km, T)
                    lo, hi = km.band(GRID)
                    for t, s_, a, b, nr, H, h in zip(GRID, km(GRID), lo, hi, at_risk(T, GRID),
                                                     na_at(T, E, GRID), smoothed_hazard(T, E, GRID)):
                        out["curves"].append({"model": m, "config": c, "benchmark": bench,
                                              "ending": "model" if mode == "censor" else "model or breaker",
                                              "t": int(t), "at_risk": int(nr), "S": round(float(s_), 6),
                                              "S_lo": round(float(a), 6), "S_hi": round(float(b), 6),
                                              "H": round(float(H), 6),
                                              "hazard_per_1k": None if np.isnan(h) else round(float(1000 * h), 6)})
                out["endings"][f"{m}/{c}/{bench}"] = ending_counts(rs[m])
            # Shapley decomposition
            fp = {m: factors(st[m]) for m in MODELS}
            fb = {m: factors(st[m], w) for m in MODELS}
            sp, sb = shapley(fp["base"], fp["p50c"]), shapley(fb["base"], fb["p50c"])
            out["decomposition"][key] = {
                "factors": {m: {k: scalar(v) for k, v in fp[m].items()} for m in MODELS},
                **{k: {"estimate": scalar(sp[k]), **ci(sb[k])} for k in sp}}
            # paired cells
            ps = pair_stats(rs["base"], rs["p50c"], ids)
            cp, cb = cells_of(ps), cells_of(ps, w)
            out["cells"][key] = {"pairs": int(ps["pairs"].sum()), **{
                cell: {k: {"estimate": scalar(cp[cell][k]), **ci(cb[cell][k])} for k in cp[cell]}
                for cell in CELLS + (ONE_FINISHED,)}}
            # problems where neither model was cut
            common = neither_cut(rs["base"], rs["p50c"], ids)
            if common:
                sub = [ids.index(i) for i in common]
                acc = {m: ratio(st[m]["cf"][sub] + st[m]["cc"][sub], st[m]["n"][sub]) for m in MODELS}
                wc = boot_weights(len(common), n_boot)
                d = wc @ (acc["p50c"] - acc["base"]) / len(common)
                out["neither_cut"][key] = {"problems": len(common),
                                           **{m: float(acc[m].mean()) for m in MODELS},
                                           "delta": {"estimate": float((acc["p50c"] - acc["base"]).mean()), **ci(d)}}
            else:
                out["neither_cut"][key] = {"problems": 0}
            # budget curve
            pts = {}
            for L in BUDGETS:
                k = int(np.searchsorted(GRID, L))
                v = {m: st[m]["within"][:, k] for m in MODELS}
                est = {m: float(v[m].sum() / st[m]["n"].sum()) for m in MODELS}
                bs = {m: (w @ v[m]) / (w @ st[m]["n"]) for m in MODELS}
                pts[str(L)] = {**{m: {"estimate": est[m], **ci(bs[m])} for m in MODELS},
                               "delta": {"estimate": est["p50c"] - est["base"], **ci(bs["p50c"] - bs["base"])}}
            out["budget"][key] = {"points": pts, **{
                f"acc_{m}": float((st[m]["cf"].sum() + st[m]["cc"].sum()) / st[m]["n"].sum()) for m in MODELS}}
            for m in MODELS:
                acc_L = st[m]["within"].sum(axis=0) / st[m]["n"].sum()
                out["budget_curves"] += [{"model": m, "config": c, "benchmark": bench, "L": int(L),
                                          "acc": round(float(a), 6)}
                                         for L, a in zip(GRID, acc_L)]
    return out


def d0_d3_identity(recs) -> dict:
    """D3's thinking is D0's request with the breaker on top, with the same seed per sample. Counts how many D3
    samples (MATH-500, sample 0, not cut by the breaker) have the same thinking and output length as D0's sample 0:
    vLLM need not reproduce a seeded sample exactly inside a different batch, so the two are equal in distribution
    only."""
    g = group(recs)
    out = {}
    for m in MODELS:
        if (m, "D0", "math500") not in g or (m, "D3", "math500") not in g:
            continue
        d0 = {r["problem_id"]: r for r in g[(m, "D0", "math500")] if r["sample_idx"] == 0}
        same = n = 0
        for r in g[(m, "D3", "math500")]:
            if r["sample_idx"] == 0 and not r["loop_triggered"] and r["problem_id"] in d0:
                n += 1
                q = d0[r["problem_id"]]
                same += r["think_tokens"] == q["think_tokens"] and r["n_gen_tokens"] == q["n_gen_tokens"]
        out[m] = {"compared": n, "identical_lengths": same}
    return out


# ---------------------------------------------------------------- report
def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def pp(x) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f}"


def ci_pp(e: dict) -> str:
    c = e.get("ci95")
    if e.get("estimate") is None or c is None:
        return "n/a"
    s = f"{pp(e['estimate'])} [{pp(c[0])}, {pp(c[1])}]"
    return s + (f" ({e['undefined_replicates']} undefined)" if e.get("undefined_replicates") else "")


def markdown(res: dict, meta: dict) -> str:
    L = ["# RQ3 results: survival and decomposition", "",
         f"Computed by `python -m wsprev.survival` on {meta['time']} (git {meta['git']}) from the records of the GPU "
         "run (`runs/gpushare_2026-11-04/runs_records.tgz`, unpacked into `runs/`). Definitions: the docstring of "
         "`src/wsprev/survival.py`. Intervals: 95% bootstrap over problems "
         f"({N_BOOT:,} resamples, `default_rng(0)`, problems in sorted order). Percentage points (pp) are signed.", "",
         "## How the thinking ended", "",
         "| Model | Config | Benchmark | Samples | `</think>` | end of text, no `</think>` | breaker | cut |",
         "|---|---|---|---|---|---|---|---|"]
    for key, c in res["endings"].items():
        m, cfg, b = key.split("/")
        n = sum(c.values())
        L.append(f"| {m} | {cfg} | {b} | {n} | {c.get('</think>', 0)} | {c.get('end of text without </think>', 0)} | "
                 f"{c.get('breaker', 0)} | {c.get('cut', 0)} |" if cfg != "D2a" and cfg != "D2b" else
                 f"| {m} | {cfg} | {b} | {n} | (thinking off) | {c.get('end of text', 0)} | | {c.get('cut', 0)} |")
    L += ["", "## Kaplan–Meier: share still thinking, $\\hat S(t)$", "",
          "Thinking on: $T$ = thinking tokens, censored at the token limit (D0, D1) or at the breaker (D3); "
          "\"/breaker-as-end\" counts the breaker's cut as an end. Thinking off: $T$ = answer tokens. "
          "RMST = $\\int_0^{32768} \\hat S(t)\\,dt$ (tokens).", "",
          "| Model / config / benchmark | Samples | Events | " + " | ".join(f"$\\hat S({t})$" for t in S_POINTS)
          + " | Median | RMST |", "|---|---|---|" + "---|" * len(S_POINTS) + "---|---|"]
    for key, k in res["km"].items():
        med = "not reached" if k["median"] is None else f"{k['median']:,.0f}"
        L.append(f"| {key} | {k['samples']} | {k['events']} | "
                 + " | ".join(pct(k["S"][str(t)]) for t in S_POINTS) + f" | {med} | {k['rmst']:,.0f} |")
    L += ["", "## Shapley decomposition of the edit effect", "",
          "$\\mathrm{Acc} = p\\,a_F + (1-p)\\,a_C$ with $p$ = share finished (not cut), $a_F$, $a_C$ = accuracy of "
          "finished and of cut answers. Termination $= (p_1-p_0)\\,[(a_{F0}-a_{C0})+(a_{F1}-a_{C1})]/2$; "
          "conditional $= \\bar p\\,(a_{F1}-a_{F0}) + (1-\\bar p)\\,(a_{C1}-a_{C0})$; they add up to $\\Delta$. "
          "Index 0 = base, 1 = p50c.", "",
          "| Config / benchmark | $p_0$ | $p_1$ | $a_{F0}$ | $a_{F1}$ | $\\Delta$ (pp) | Termination (pp) | "
          "Conditional (pp) |", "|---|---|---|---|---|---|---|---|"]
    for key, d in res["decomposition"].items():
        f0, f1 = d["factors"]["base"], d["factors"]["p50c"]
        L.append(f"| {key} | {pct(f0['p'])} | {pct(f1['p'])} | {pct(f0['aF'])} | {pct(f1['aF'])} | "
                 f"{pp(d['delta']['estimate'])} | {ci_pp(d['termination'])} | {ci_pp(d['conditional'])} |")
    L += ["", "## Paired cells (sample $s$ of base with sample $s$ of p50c)", "",
          "$\\Delta = \\sum_{\\text{cells}} P(\\text{cell})\\,[\\mathrm{Acc}_1 - \\mathrm{Acc}_0 \\mid \\text{cell}]$. "
          "\"Both finished\" compares the models on the same pairs.", "",
          "| Config / benchmark | Pairs | Cell | Share | Acc base | Acc p50c | Contribution (pp) |",
          "|---|---|---|---|---|---|---|"]
    for key, d in res["cells"].items():
        for cell in CELLS + (ONE_FINISHED,):
            e = d[cell]
            L.append(f"| {key} | {d['pairs']} | {cell} | {pct(e['share']['estimate'])} | "
                     f"{pct(e['acc_base']['estimate'])} | {pct(e['acc_p50c']['estimate'])} | "
                     f"{ci_pp(e['contribution'])} |")
    L += ["", "## Problems where neither model was cut", "",
          "| Config / benchmark | Problems | base | p50c | $\\Delta$ (pp) |", "|---|---|---|---|---|"]
    for key, d in res["neither_cut"].items():
        if d["problems"]:
            L.append(f"| {key} | {d['problems']} | {pct(d['base'])} | {pct(d['p50c'])} | {ci_pp(d['delta'])} |")
        else:
            L.append(f"| {key} | 0 | | | |")
    L += ["", "## Budget curve: correct and finished within $L$ generated tokens", "",
          "$\\mathrm{Acc}(L) = P(\\text{correct}, \\text{finished}, N \\le L)$, $N$ = generated tokens. "
          "\"Acc\" is the run's accuracy, which also counts cut answers graded correct.", "",
          "| Config / benchmark | Model | " + " | ".join(f"$L = {b}$" for b in BUDGETS) + " | Acc |",
          "|---|---|" + "---|" * len(BUDGETS) + "---|"]
    for key, d in res["budget"].items():
        for m in MODELS:
            L.append(f"| {key} | {m} | " + " | ".join(pct(d["points"][str(b)][m]["estimate"]) for b in BUDGETS)
                     + f" | {pct(d[f'acc_{m}'])} |")
        L.append(f"| {key} | $\\Delta$ (pp) | " + " | ".join(ci_pp(d["points"][str(b)]["delta"]) for b in BUDGETS)
                 + " | |")
    ident = res.get("d0_d3_identity", {})
    if ident:
        L += ["", "## D3 and D0 are equal in distribution, not sample by sample", ""]
        for m, v in ident.items():
            L.append(f"- {m}: of {v['compared']} D3 samples on MATH-500 with seed 0 that the breaker did not cut, "
                     f"{v['identical_lengths']} have the same thinking and output length as D0's seed-0 sample.")
    return "\n".join(L) + "\n"


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.environ.get("WSPREV_RUNS", str(REPO / "runs")))
    ap.add_argument("--out", default=str(REPO / "results" / "rq3"))
    ap.add_argument("--summary", default=str(REPO / "results" / "rq3_summary.md"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    recs = read_records([Path(a.runs) / "d0"])[0] + read_records([Path(a.runs) / "rq1"])[0]
    res = analyse(recs, a.n_boot)
    res["d0_d3_identity"] = d0_d3_identity(recs)
    meta = {"time": time.strftime("%Y-%m-%d %H:%M"), "n_boot": a.n_boot, "boot_seed": BOOT_SEED,
            "bandwidth": BANDWIDTH, "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                                          capture_output=True, text=True).stdout.strip() or None,
            "records": len(recs)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    curves, budget_curves = res.pop("curves"), res.pop("budget_curves")
    (out / "rq3.json").write_text(json.dumps({"meta": meta, **res}, indent=1) + "\n", encoding="utf-8")
    write_csv(out / "km_curves.csv", curves)
    write_csv(out / "budget_curves.csv", budget_curves)
    report = markdown(res, meta)
    Path(a.summary).write_text(report, encoding="utf-8")
    print(report)
    print(f"-> {out / 'rq3.json'}, {out / 'km_curves.csv'}, {out / 'budget_curves.csv'}, {a.summary}")


if __name__ == "__main__":
    main()
