"""Figures of RQ3 from the small result files of `python -m wsprev.survival` (results/rq3/*.json, *.csv); the raw
records are not needed.

    python scripts/plot_rq3.py [--rq3 results/rq3]       (needs matplotlib)

Writes results/rq3/figs/{survival,survival_d3,hazard,decomposition,budget}.{pdf,png}:
- survival:      Kaplan-Meier share still thinking, base vs p50c, MATH-500, D0 and D1 (paper figure 1)
- survival_d3:   the same under D3, the breaker as censoring vs as an end, next to D0 (appendix)
- hazard:        smoothed Nelson-Aalen hazard of ending the thinking, D0 and D1 (appendix)
- decomposition: Shapley decomposition and paired cells of the edit effect, MATH-500 (paper figure 3)
- budget:        Acc(L), correct and finished within L tokens, MATH-500 (paper figure 4)

Colours (validated with the dataviz skill's palette checker): base = blue, p50c = orange in every figure; the two
components of a decomposition = aqua and violet, with value labels (aqua is below 3:1 contrast on white).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODEL_COLOR = {"base": "#2a78d6", "p50c": "#eb6834"}
PART_COLOR = ("#1baf7a", "#6250d6")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
L_MAX = 32768

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "savefig.dpi": 200, "pdf.fonttype": 42,
})


def ktok(x, _pos=None):
    return "0" if x == 0 else f"{x / 1024:g}k"


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def curves(rows, **where):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if all(r[k] == v for k, v in where.items()):
            for k, v in r.items():
                out[r["model"]][k].append(v)
    return out


def num(xs):
    return [float("nan") if x in ("", "None") else float(x) for x in xs]


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out / name}.pdf/.png")


def time_axis(ax, label="tokens generated"):
    ax.set_xlim(0, L_MAX)
    ax.set_xticks(range(0, L_MAX + 1, 8192))
    ax.xaxis.set_major_formatter(FuncFormatter(ktok))
    ax.set_xlabel(label)


def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=8.5, color=INK2)


def plot_survival(km, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, cfg in zip(axes, ("D0", "D1")):
        c = curves(km, config=cfg, benchmark="math500", ending="model")
        for m in ("base", "p50c"):
            t, s, lo, hi = (num(c[m][k]) for k in ("t", "S", "S_lo", "S_hi"))
            ax.fill_between(t, lo, hi, step="post", color=MODEL_COLOR[m], alpha=0.12, linewidth=0)
            ax.step(t, s, where="post", color=MODEL_COLOR[m], label=m)
            end_label(ax, t[-1], s[-1], m)
        ax.set_title(f"{cfg}, MATH-500", loc="left")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        time_axis(ax, "thinking tokens $t$")
    axes[0].set_ylabel(r"still thinking, $\hat S(t)$")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, out, "survival")


def plot_survival_d3(km, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    styles = (("D0", "model", "-", "D0"), ("D3", "model", "--", "D3, breaker censors"),
              ("D3", "model or breaker", ":", "D3, breaker ends"))
    for ax, m in zip(axes, ("base", "p50c")):
        for cfg, ending, ls, lab in styles:
            c = curves(km, config=cfg, benchmark="math500", ending=ending)[m]
            ax.step(num(c["t"]), num(c["S"]), where="post", color=MODEL_COLOR[m], linestyle=ls, label=lab)
        ax.set_title(f"{m}, MATH-500", loc="left")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        time_axis(ax, "thinking tokens $t$")
        ax.legend(loc="lower left")
    axes[0].set_ylabel(r"still thinking, $\hat S(t)$")
    fig.tight_layout()
    save(fig, out, "survival_d3")


def plot_hazard(km, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, cfg in zip(axes, ("D0", "D1")):
        c = curves(km, config=cfg, benchmark="math500", ending="model")
        for m in ("base", "p50c"):
            ax.plot(num(c[m]["t"]), num(c[m]["hazard_per_1k"]), color=MODEL_COLOR[m], label=m)
        ax.set_title(f"{cfg}, MATH-500", loc="left")
        time_axis(ax, "thinking tokens $t$")
    axes[0].set_ylabel("hazard of ending\n(per 1,000 tokens)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, "hazard")


def hbars(ax, rows, parts, labels, title):
    """Horizontal grouped bars: one group per config, one bar per part, and the total as a black tick."""
    h = 0.32
    for g, (cfg, d) in enumerate(rows):
        for j, (p, lab, col) in enumerate(zip(parts, labels, PART_COLOR)):
            e = d[p]
            y = g + (j - 0.5) * (h + 0.04)
            v = 100 * e["estimate"]
            lo, hi = (100 * x for x in e["ci95"])
            ax.barh(y, v, height=h, color=col, label=lab if g == 0 else None)
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
            ax.annotate(f"{v:+.1f}", (hi if v >= 0 else lo, y), xytext=(3 if v >= 0 else -3, 0),
                        textcoords="offset points", ha="left" if v >= 0 else "right", va="center",
                        fontsize=7.5, color=INK2)
        tot = 100 * d["total"]
        ax.plot([tot, tot], [g - 0.42, g + 0.42], color=INK, linewidth=1.6, solid_capstyle="butt",
                label=r"total $\Delta$" if g == 0 else None)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([c for c, _ in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)          # first config on top (set, not inverted: the axes share y)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("percentage points")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=8, ncol=1)


def plot_decomposition(res, out):
    cfgs = ("D0", "D1", "D3", "D2b")
    shap = [(c, {**res["decomposition"][f"{c}/math500"],
                 "total": res["decomposition"][f"{c}/math500"]["delta"]["estimate"]}) for c in cfgs]
    cells = [(c, {"one": res["cells"][f"{c}/math500"]["one finished"]["contribution"],
                  "both": res["cells"][f"{c}/math500"]["both finished"]["contribution"],
                  "total": res["decomposition"][f"{c}/math500"]["delta"]["estimate"]}) for c in cfgs]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    hbars(axes[0], shap, ("termination", "conditional"), ("termination", "conditional accuracy"),
          "(a) Shapley decomposition")
    hbars(axes[1], cells, ("one", "both"), ("pairs where one model finished", "pairs where both finished"),
          "(b) paired cells")
    lo = min(a.get_xlim()[0] for a in axes)
    hi = max(a.get_xlim()[1] for a in axes)
    for a in axes:
        a.set_xlim(lo - 5, hi + 6)
    fig.suptitle(r"Edit effect $\Delta$ on MATH-500 (p50c $-$ base), 95% bootstrap intervals", x=0.01, ha="left",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    save(fig, out, "decomposition")


def plot_budget(budget, out):
    cfgs = ("D0", "D1", "D3", "D2b")
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.6), sharey=True)
    for ax, cfg in zip(axes, cfgs):
        c = curves(budget, config=cfg, benchmark="math500")
        for m in ("base", "p50c"):
            L, a = num(c[m]["L"]), num(c[m]["acc"])
            keep = [i for i, x in enumerate(L) if x >= 1024]
            ax.plot([L[i] for i in keep], [a[i] for i in keep], color=MODEL_COLOR[m], label=m)
            ax.plot(L[-1], a[-1], "o", ms=5, color=MODEL_COLOR[m], mec="white", mew=1.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 4096, 16384, 32768])
        ax.xaxis.set_major_formatter(FuncFormatter(ktok))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"{cfg}", loc="left")
        ax.set_xlabel("budget $L$ (tokens)")
        ax.set_ylim(0, 0.62)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("correct within $L$")
    axes[0].legend(loc="upper left")
    fig.suptitle("MATH-500: share of samples ending with a correct answer within $L$ generated tokens", x=0.01,
                 ha="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    save(fig, out, "budget")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq3", default=str(REPO / "results" / "rq3"))
    a = ap.parse_args(argv)
    d = Path(a.rq3)
    out = d / "figs"
    out.mkdir(parents=True, exist_ok=True)
    km, budget = read_csv(d / "km_curves.csv"), read_csv(d / "budget_curves.csv")
    res = json.loads((d / "rq3.json").read_text(encoding="utf-8"))
    plot_survival(km, out)
    plot_survival_d3(km, out)
    plot_hazard(km, out)
    plot_decomposition(res, out)
    plot_budget(budget, out)


if __name__ == "__main__":
    main()
