"""RQ1 analysis, exactly as registered in prereg/criteria.md (sections 3-6), and the two choices made on practice
problems before the benchmark runs (D3's tau; the base configuration of the cross comparison).

    python -m wsprev.rq1 [--runs $WSPREV_RUNS] [--out results/rq1]

Reads $WSPREV_RUNS/d0 (the D0 calibration run, MATH-500, 1 sample) and $WSPREV_RUNS/rq1 (D1, D2a, D2b, D3; 2 samples
on MATH-500), and the original evaluation's per-problem results ($WSPREV_LOGS/exp8_math/eval) for D0 on GSM8K and
IFEval; writes results/rq1/rq1.json and results/rq1_summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data, decoding
from ._vendor import REPO
from .summary import authors_per_problem, group, mean, metrics, pct, read_records

MODELS = ("base", "p50c")
BENCHES = ("math500", "gsm8k", "ifeval")
H_CONFIGS = ("D0", "D1", "D2b", "D3")      # configurations in the hypotheses
RQ1_CONFIGS = ("D1", "D2a", "D2b", "D3")   # benchmark runs of RQ1 (D0 is the calibration run)
SELECT_CONFIGS = ("D1", "D2b", "D3")       # candidates for the cross comparison, in tie-break order
EXPECTED_SAMPLES = {"D0": 1, "D1": 2, "D2a": 2, "D2b": 2, "D3": 2}   # MATH-500
ORIGINAL_EVAL = {"base": "unedited", "p50c": "p50c"}   # folders of exp8_math/eval/ in the authors' logs
H1_THRESHOLD, MARGIN, ALPHA, N_BOOT, BOOT_SEED = 0.30, 0.05, 0.05, 10_000, 0
TAU_PROBLEMS = 256                         # the first 256 practice DEV problems
CONCLUSIONS = {   # prereg section 6, verbatim
    "termination": "The edit's gain on MATH-500 mainly reflects the unedited model's failure to stop thinking; "
                   "decoding-level remedies give the unedited model most of it.",
    "edit": "The edit retains a substantial gain under the fair baselines we tested.",
    "mixed": None,   # the effect sizes are reported as they are, without either summary sentence
}


# ---------------------------------------------------------------- choices on practice problems
def choose_tau(correct: dict[float, int]) -> float:
    """tau with the most correct answers (same problems for every tau); ties toward the larger tau."""
    best = max(correct.values())
    return max(t for t, c in correct.items() if c == best)


def choose_config(correct: dict[str, int]) -> str:
    """The base configuration with the most correct answers on SELECT; ties: D1, then D2b, then D3."""
    best = max(correct.values())
    return next(c for c in SELECT_CONFIGS if correct.get(c) == best)


# ---------------------------------------------------------------- the registered tests
def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values."""
    order = sorted(p, key=p.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p[k]))
        adj[k] = running
    return adj


def problem_accuracy(recs, ids, samples: int) -> np.ndarray:
    """Accuracy per problem (mean over its samples) in the order of `ids`; every problem needs `samples` samples."""
    per = defaultdict(list)
    for r in recs:
        per[r["problem_id"]].append(r["correct"])
    short = [i for i in ids if len(per.get(i, ())) < samples]
    if short:
        raise ValueError(f"{len(short)} problems have fewer than {samples} samples, e.g. {short[:3]}")
    return np.array([np.mean(per[i]) for i in ids], dtype=float)


def analyse(acc: dict[tuple[str, str], np.ndarray], cross: str | None = None) -> dict:
    """acc[(model, config)]: per-problem MATH-500 accuracy, the same problem order in every cell."""
    n = len(next(iter(acc.values())))
    idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))   # one resampling for every cell
    boot = {k: v[idx].mean(axis=1) for k, v in acc.items()}
    point = {k: float(v.mean()) for k, v in acc.items()}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    delta = {c: point[("p50c", c)] - point[("base", c)] for c in H_CONFIGS}
    dboot = {c: boot[("p50c", c)] - boot[("base", c)] for c in H_CONFIGS}
    p = {"H2": float(np.mean(dboot["D0"] - dboot["D1"] <= 0)),
         "H3": float(np.mean(dboot["D3"] >= MARGIN)),
         "H4": float(np.mean(dboot["D2b"] >= MARGIN))}
    adj = holm(p)
    res = {"n_problems": n, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "margin": MARGIN, "alpha": ALPHA,
           "accuracy": {f"{m}/{c}": point[(m, c)] for m, c in sorted(acc)},
           "delta": {c: {"estimate": delta[c], "ci95": ci(dboot[c])} for c in H_CONFIGS},
           "H1": {"estimate": delta["D0"], "ci95": ci(dboot["D0"]), "threshold": H1_THRESHOLD,
                  "supported": bool(delta["D0"] >= H1_THRESHOLD)},
           "H2": {"estimate": delta["D0"] - delta["D1"], "ci95": ci(dboot["D0"] - dboot["D1"])},
           "H3": {"estimate": delta["D3"], "ci95": ci(dboot["D3"])},
           "H4": {"estimate": delta["D2b"], "ci95": ci(dboot["D2b"])}}
    for h in ("H2", "H3", "H4"):
        res[h].update({"p": p[h], "p_holm": adj[h], "supported": bool(adj[h] < ALPHA)})
    if cross:
        x = boot[("p50c", "D0")] - boot[("base", cross)]
        res["cross"] = {"base_config": cross, "estimate": point[("p50c", "D0")] - point[("base", cross)],
                        "ci95": ci(x)}
    s3, s4 = res["H3"]["supported"], res["H4"]["supported"]
    res["conclusion"] = "termination" if s3 and s4 else "edit" if not (s3 or s4) else "mixed"
    res["conclusion_text"] = CONCLUSIONS[res["conclusion"]]
    return res


# ---------------------------------------------------------------- reported without tests
def expected_samples(bench: str, config: str) -> int:
    """Samples per problem (prereg section 4): MATH-500 2 (D0: 1); GSM8K 1; IFEval 2."""
    return EXPECTED_SAMPLES[config] if bench == "math500" else data.RQ1_SAMPLES[bench]


def effect_ci(base: np.ndarray, p50c: np.ndarray) -> dict:
    """Edit effect on the same problems with the 95% percentile interval of the paired bootstrap: the resampling of
    analyse (default_rng(BOOT_SEED), N_BOOT draws of problem indices)."""
    n = len(base)
    idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))
    d = p50c[idx].mean(axis=1) - base[idx].mean(axis=1)
    return {"problems": n, "base": float(base.mean()), "p50c": float(p50c.mean()),
            "estimate": float(p50c.mean() - base.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def secondary_effects(groups, bench_ids: dict, original_eval=None) -> dict:
    """Edit effect with its 95% CI for every configuration on every benchmark (prereg section 5, without tests).
    D0 on GSM8K and IFEval uses the original evaluation's per-problem results under <original_eval>, the D0 data that
    prereg section 4 names. A cell whose runs are incomplete is reported as such."""
    out = {}
    for bench, ids in bench_ids.items():
        for c in ("D0",) + RQ1_CONFIGS:
            cells, source = {}, "this study"
            try:
                for m in MODELS:
                    if (m, c, bench) in groups:
                        cells[m] = problem_accuracy(groups[(m, c, bench)], ids, expected_samples(bench, c))
                    elif c == "D0" and bench != "math500" and original_eval is not None:
                        per = authors_per_problem(Path(original_eval) / ORIGINAL_EVAL[m], bench)
                        if not per:
                            continue
                        missing = [i for i in ids if i not in per]
                        if missing:
                            raise ValueError(f"the original evaluation lacks {len(missing)} problems, e.g. {missing[:3]}")
                        cells[m] = np.array([np.mean(per[i]) for i in ids], dtype=float)
                        source = "original evaluation"
            except ValueError as e:
                out[f"{c}/{bench}"] = {"incomplete": str(e)}
                continue
            if len(cells) == 2:
                out[f"{c}/{bench}"] = dict(effect_ci(cells["base"], cells["p50c"]), source=source)
            elif cells:
                out[f"{c}/{bench}"] = {"incomplete": f"no runs of {sorted(set(MODELS) - set(cells))}"}
    return out


def both_finished(groups) -> dict:
    """Per configuration and benchmark: both models' accuracy on the problems where no sample of either was cut."""
    out = {}
    for (m, c, b), rs in sorted(groups.items()):
        if m != "base" or ("p50c", c, b) not in groups:
            continue
        per = {k: defaultdict(list) for k in MODELS}
        for model in MODELS:
            for r in groups[(model, c, b)]:
                per[model][r["problem_id"]].append(r)
        common = [i for i in per["base"] if i in per["p50c"]
                  and not any(r["cut"] for model in MODELS for r in per[model][i])]
        out[f"{c}/{b}"] = {"problems": len(common), **{
            model: mean(r["correct"] for i in common for r in per[model][i]) for model in MODELS}}
    return out


# ---------------------------------------------------------------- report (formulas in LaTeX, as in the prereg)
def tex_delta(c: str) -> str:
    return r"\Delta_{\mathrm{" + c + "}}"


def pp(x: float) -> str:
    return f"{100 * x:.1f}"


def markdown(res: dict, desc: dict, fin: dict, sec: dict | None = None) -> str:
    L = ["# RQ1 results", "",
         f"Registered analysis of `prereg/criteria.md`, computed by `python -m wsprev.rq1` on "
         f"{time.strftime('%Y-%m-%d %H:%M')} (git {res.get('git')}). D3 threshold $\\tau = {res.get('tau')}$.", "",
         f"## Edit effect on MATH-500 ({res['n_problems']} problems; 95% bootstrap CIs, {res['n_boot']:,} resamples)",
         "", "| Configuration | base | p50c | $\\Delta$ (pp) | 95% CI (pp) |", "|---|---|---|---|---|"]
    for c in H_CONFIGS:
        d = res["delta"][c]
        L.append(f"| {c} | {pct(res['accuracy'][f'base/{c}'])} | {pct(res['accuracy'][f'p50c/{c}'])} | "
                 f"{pp(d['estimate'])} | [{pp(d['ci95'][0])}, {pp(d['ci95'][1])}] |")
    statements = {"H1": f"${tex_delta('D0')} \\ge 0.30$", "H2": f"${tex_delta('D1')} \\lt {tex_delta('D0')}$",
                  "H3": f"${tex_delta('D3')} \\lt m$", "H4": f"${tex_delta('D2b')} \\lt m$"}
    L += ["", f"## Hypotheses (margin $m = {MARGIN}$, Holm over H2–H4 at $\\alpha = {ALPHA}$)", "",
          "| | Statement | Estimate (pp) | $p$ | $p$ (Holm) | Supported |", "|---|---|---|---|---|---|",
          f"| H1 | {statements['H1']} | {pp(res['H1']['estimate'])} | | | {'yes' if res['H1']['supported'] else 'no'} |"]
    for h in ("H2", "H3", "H4"):
        r = res[h]
        L.append(f"| {h} | {statements[h]} | {pp(r['estimate'])} | {r['p']:.4f} | {r['p_holm']:.4f} | "
                 f"{'yes' if r['supported'] else 'no'} |")
    L += ["", "## Conclusion", "", res["conclusion_text"] or
          "H3 and H4 do not agree: the effect sizes above are reported as they are (prereg section 6).", ""]
    if "cross" in res:
        x, c = res["cross"], res["cross"]["base_config"]
        L += ["## Cross comparison", "",
              f"$\\operatorname{{Acc}}(\\mathrm{{p50c}}, \\mathrm{{D0}}) - \\operatorname{{Acc}}(\\mathrm{{base}}, "
              f"\\mathrm{{{c}}}) = {pp(x['estimate'])}$ pp (95% CI [{pp(x['ci95'][0])}, {pp(x['ci95'][1])}]); "
              f"{c} was chosen on the practice SELECT split.", ""]
    if sec:
        L += ["## Edit effect on every benchmark (secondary, no tests)", "",
              "| Benchmark | Configuration | Problems | base | p50c | $\\Delta$ (pp) | 95% CI (pp) | Data |",
              "|---|---|---|---|---|---|---|---|"]
        for key, e in sec.items():
            c, b = key.split("/")
            if "incomplete" in e:
                L.append(f"| {b} | {c} | | | | | | incomplete: {e['incomplete']} |")
            else:
                L.append(f"| {b} | {c} | {e['problems']} | {pct(e['base'])} | {pct(e['p50c'])} | {pp(e['estimate'])} "
                         f"| [{pp(e['ci95'][0])}, {pp(e['ci95'][1])}] | {e['source']} |")
        L.append("")
    L += ["## Every run", "", "| Model | Config | Benchmark | Problems | Samples | Accuracy | Cut | Mean tokens | "
          "Acc. finished | Acc. cut | Breaker |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, m in desc.items():
        model, config, bench = key.split("/")
        L.append(f"| {model} | {config} | {bench} | {m['problems']} | {m['samples']} | {pct(m['accuracy'])} | "
                 f"{pct(m['cut_share'])} | {m['mean_tokens']:,.0f} | {pct(m['accuracy_finished'])} | "
                 f"{pct(m['accuracy_cut'])} | {pct(m['breaker_share'])} |")
    L += ["", "## Problems where neither model was cut", "", "| Config / benchmark | Problems | base | p50c |",
          "|---|---|---|---|"]
    L += [f"| {k} | {v['problems']} | {pct(v['base'])} | {pct(v['p50c'])} |" for k, v in fin.items()]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.environ.get("WSPREV_RUNS", str(data.home() / "runs")))
    ap.add_argument("--out", default=str(REPO / "results" / "rq1"))
    a = ap.parse_args(argv)
    if decoding.D3_TAU is None:
        raise SystemExit("decoding.D3_TAU is not set")
    recs = read_records([Path(a.runs) / "d0"])[0] + read_records([Path(a.runs) / "rq1"])[0]
    groups = group(recs)
    bench_ids = {b: [r["id"] for r in data.load(b)] for b in BENCHES}
    acc = {(m, c): problem_accuracy(groups[(m, c, "math500")], bench_ids["math500"], EXPECTED_SAMPLES[c])
           for m in MODELS for c in H_CONFIGS}
    choice_path = REPO / "results" / "select" / "choice.json"
    cross = json.loads(choice_path.read_text(encoding="utf-8"))["choice"] if choice_path.exists() else None
    res = analyse(acc, cross)
    res["tau"] = decoding.D3_TAU
    res["git"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or None
    sec = secondary_effects(groups, bench_ids, data.logs_dir() / "exp8_math" / "eval")
    desc = {f"{m}/{c}/{b}": metrics(rs) for (m, c, b), rs in sorted(groups.items())}
    fin = both_finished(groups)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rq1.json").write_text(json.dumps({"tests": res, "secondary": sec, "runs": desc, "both_finished": fin},
                                             indent=1) + "\n", encoding="utf-8")
    report = markdown(res, desc, fin, sec)
    (REPO / "results" / "rq1_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
