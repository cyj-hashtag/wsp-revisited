"""Summaries of generation runs (output folders of wsprev.generate).

    python scripts/summarize.py <folder> [<folder> ...] [--authors <logs>/exp8_math/eval/unedited] [--gate base|p50c]
                                [--out results/<name>]

A folder holds records.jsonl itself or in sub-folders (shards). Records are grouped by (model, config, benchmark).
Per group: accuracy over all samples (with its standard error over problems), share cut at the token cap, mean
generated tokens, accuracy of finished and of cut answers, share of thinking closed by the model, breaker firing
share and mean position, tokens/s (meta.json).
--authors  the original evaluation's accuracy on the same problems, all of its samples
           (<eval>/parts/*/<benchmark>_per_problem.json).
--gate     the D0 calibration interval of docs/plan-for-agents.md section 3 for that model (MATH-500).
--out      writes summary.json and per_problem.csv there (small files that go into git).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

GATES = {"base": {"acc": (0.045, 0.115), "cut_about": 0.93}, "p50c": {"acc": (0.43, 0.55), "cut_about": 0.33}}


def read_records(folders):
    recs, metas = [], []
    for folder in folders:
        paths = sorted(glob.glob(os.path.join(folder, "**", "records.jsonl"), recursive=True))
        if not paths:
            sys.exit(f"no records.jsonl under {folder}")
        for p in paths:
            with open(p, encoding="utf-8") as f:
                recs += [json.loads(line) for line in f if line.strip()]
            meta = Path(p).with_name("meta.json")
            if meta.exists():
                metas.append(json.loads(meta.read_text(encoding="utf-8")))
    return recs, metas


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def se_over_problems(recs):
    """Standard error of the accuracy, treating problems as the sampling unit."""
    per = defaultdict(list)
    for r in recs:
        per[r["problem_id"]].append(r["correct"])
    accs = [mean(v) for v in per.values()]
    if len(accs) < 2:
        return None
    m = mean(accs)
    return math.sqrt(sum((a - m) ** 2 for a in accs) / (len(accs) - 1) / len(accs))


def metrics(recs):
    fin = [r for r in recs if not r["cut"]]
    cut = [r for r in recs if r["cut"]]
    thinking = [r for r in recs if r.get("think_closed") is not None]
    fired = [r for r in recs if r.get("loop_triggered")]
    return {
        "problems": len({r["problem_id"] for r in recs}), "samples": len(recs),
        "accuracy": mean(r["correct"] for r in recs), "accuracy_se": se_over_problems(recs),
        "cut_share": mean(r["cut"] for r in recs), "mean_tokens": mean(r["n_gen_tokens"] for r in recs),
        "accuracy_finished": mean(r["correct"] for r in fin), "accuracy_cut": mean(r["correct"] for r in cut),
        "think_closed_share": mean(r["think_closed"] for r in thinking) if thinking else None,
        "breaker_share": mean(bool(r.get("loop_triggered")) for r in recs),
        "breaker_mean_pos": mean(r["trigger_pos"] for r in fired) if fired else None,
        "no_answer_share": mean(r.get("pred") is None for r in recs if "n_boxes" in r),
        "mean_dup8_tail": mean(r["dup8_tail"] for r in recs),
        "grade_errors": sum("grade_error" in r for r in recs),
    }


def authors_accuracy(eval_dir, bench, problem_ids):
    per = {}
    for p in glob.glob(os.path.join(eval_dir, "parts", "*", f"{bench}_per_problem.json")):
        for row in json.load(open(p, encoding="utf-8")):
            per[row["id"]] = row["correct"]
    common = [i for i in problem_ids if i in per]
    if not common:
        return None
    flat = [c for i in common for c in per[i]]
    return {"problems": len(common), "samples": len(flat), "accuracy": mean(flat),
            "accuracy_first_sample": mean(per[i][0] for i in common)}


def throughput(metas):
    tokens = sum(s.get("tokens") or 0 for m in metas for s in m["sessions"])
    seconds = sum(s.get("seconds") or 0 for m in metas for s in m["sessions"])
    return {"tokens": tokens, "seconds": seconds, "tokens_per_s": tokens / seconds if seconds else None}


def pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--authors", help="the original evaluation folder of the same model (exp8_math/eval/<model>)")
    ap.add_argument("--gate", choices=sorted(GATES))
    ap.add_argument("--out")
    a = ap.parse_args()
    recs, metas = read_records(a.folders)
    groups = defaultdict(list)
    for r in recs:
        groups[(r["model"], r["config"], r["benchmark"])].append(r)
    summary = {"throughput": throughput(metas), "groups": {}}
    gate_ok = None
    for (model, config, bench), rs in sorted(groups.items()):
        m = metrics(rs)
        key = f"{model}/{config}/{bench}"
        print(f"\n{key}: {m['problems']} problems, {m['samples']} samples")
        print(f"  accuracy {pct(m['accuracy'])} (SE {pct(m['accuracy_se'])}) | cut at cap {pct(m['cut_share'])} | "
              f"mean tokens {m['mean_tokens']:,.0f}")
        print(f"  accuracy finished {pct(m['accuracy_finished'])} | accuracy cut {pct(m['accuracy_cut'])} | "
              f"thinking closed {pct(m['think_closed_share'])} | breaker fired {pct(m['breaker_share'])}")
        if m["grade_errors"]:
            print(f"  WARNING: {m['grade_errors']} grading errors (regrade from texts.jsonl.gz)")
        if a.authors:
            m["authors"] = authors_accuracy(a.authors, bench, sorted({r["problem_id"] for r in rs}))
            if m["authors"]:
                print(f"  original evaluation on the same {m['authors']['problems']} problems: "
                      f"{pct(m['authors']['accuracy'])} ({m['authors']['samples']} samples; first sample only "
                      f"{pct(m['authors']['accuracy_first_sample'])})")
        if a.gate and config == "D0" and bench == "math500":
            lo, hi = GATES[a.gate]["acc"]
            gate_ok = lo <= m["accuracy"] <= hi
            m["gate"] = {"interval": [lo, hi], "pass": gate_ok}
            print(f"  GATE ({a.gate}): accuracy {pct(m['accuracy'])} in [{pct(lo)}, {pct(hi)}]? "
                  f"{'PASS' if gate_ok else 'FAIL -> stop and check the evaluation pipeline'}; "
                  f"cut share {pct(m['cut_share'])} (expected about {pct(GATES[a.gate]['cut_about'])})")
        summary["groups"][key] = m
    t = summary["throughput"]
    if t["tokens_per_s"]:
        print(f"\nthroughput: {t['tokens']:,} tokens in {t['seconds'] / 3600:.2f} h = {t['tokens_per_s']:,.0f} tok/s")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        per = defaultdict(lambda: {"n": 0, "correct": 0, "cut": 0, "tokens": 0})
        for r in recs:
            c = per[(r["model"], r["config"], r["benchmark"], r["problem_id"])]
            c["n"] += 1
            c["correct"] += int(r["correct"])
            c["cut"] += int(r["cut"])
            c["tokens"] += r["n_gen_tokens"]
        with open(out / "per_problem.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "config", "benchmark", "problem_id", "samples", "correct", "cut", "mean_tokens"])
            for (model, config, bench, pid), c in sorted(per.items()):
                w.writerow([model, config, bench, pid, c["n"], c["correct"], c["cut"], round(c["tokens"] / c["n"])])
        print(f"-> {out / 'summary.json'}, {out / 'per_problem.csv'}")
    if gate_ok is False:
        sys.exit(3)


if __name__ == "__main__":
    main()
