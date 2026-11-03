"""Reading and summarizing the output folders of wsprev.generate (used by scripts/summarize.py, scripts/run_stage.py
and wsprev.rq1)."""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

# D0 calibration intervals on MATH-500 (prereg/criteria.md section 4; docs/plan-for-agents.md section 3)
GATES = {"base": {"acc": (0.045, 0.115), "cut_about": 0.93}, "p50c": {"acc": (0.43, 0.55), "cut_about": 0.33}}


def read_records(folders) -> tuple[list[dict], list[dict]]:
    """(records, meta.json contents) of every records.jsonl under the folders. A sample that occurs twice (e.g. two
    shard layouts of one job) is kept once, with a warning."""
    recs, metas, seen, dups = [], [], set(), 0
    for folder in folders:
        paths = sorted(glob.glob(os.path.join(str(folder), "**", "records.jsonl"), recursive=True))
        if not paths:
            raise FileNotFoundError(f"no records.jsonl under {folder}")
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    key = (r["model"], r["config"], r["benchmark"], r["problem_id"], r["sample_idx"])
                    if key in seen:
                        dups += 1
                        continue
                    seen.add(key)
                    recs.append(r)
            meta = Path(p).with_name("meta.json")
            if meta.exists():
                metas.append(json.loads(meta.read_text(encoding="utf-8")))
    if dups:
        print(f"WARNING: {dups} duplicate samples ignored (the same sample in two folders)")
    return recs, metas


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def se_over_problems(recs) -> float | None:
    """Standard error of the accuracy with problems as the sampling unit."""
    per = defaultdict(list)
    for r in recs:
        per[r["problem_id"]].append(r["correct"])
    accs = [mean(v) for v in per.values()]
    if len(accs) < 2:
        return None
    m = mean(accs)
    return math.sqrt(sum((a - m) ** 2 for a in accs) / (len(accs) - 1) / len(accs))


def metrics(recs) -> dict:
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


def group(recs) -> dict:
    out = defaultdict(list)
    for r in recs:
        out[(r["model"], r["config"], r["benchmark"])].append(r)
    return dict(out)


def gate(model: str, m: dict) -> dict:
    lo, hi = GATES[model]["acc"]
    return {"interval": [lo, hi], "pass": lo <= m["accuracy"] <= hi, "cut_about": GATES[model]["cut_about"]}


def authors_per_problem(eval_dir, bench) -> dict[str, list[int]]:
    """{problem id: correct (0/1 per sample)} of the original evaluation, from
    <eval_dir>/parts/*/<bench>_per_problem.json of the authors' log repository (empty if there are none)."""
    per = {}
    for p in glob.glob(os.path.join(str(eval_dir), "parts", "*", f"{bench}_per_problem.json")):
        with open(p, encoding="utf-8") as f:
            for row in json.load(f):
                per[row["id"]] = row["correct"]
    return per


def authors_accuracy(eval_dir, bench, problem_ids) -> dict | None:
    """The original evaluation's accuracy on the same problems (all of its samples)."""
    per = authors_per_problem(eval_dir, bench)
    common = [i for i in problem_ids if i in per]
    if not common:
        return None
    flat = [c for i in common for c in per[i]]
    return {"problems": len(common), "samples": len(flat), "accuracy": mean(flat),
            "accuracy_first_sample": mean(per[i][0] for i in common)}


def throughput(metas) -> dict:
    tokens = sum(s.get("tokens") or 0 for m in metas for s in m["sessions"])
    seconds = sum(s.get("seconds") or 0 for m in metas for s in m["sessions"])
    return {"tokens": tokens, "seconds": seconds, "tokens_per_s": tokens / seconds if seconds else None}


def pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def report(key: str, m: dict) -> str:
    lines = [f"{key}: {m['problems']} problems, {m['samples']} samples",
             f"  accuracy {pct(m['accuracy'])} (SE {pct(m['accuracy_se'])}) | cut at cap {pct(m['cut_share'])} | "
             f"mean tokens {m['mean_tokens']:,.0f}",
             f"  accuracy finished {pct(m['accuracy_finished'])} | accuracy cut {pct(m['accuracy_cut'])} | "
             f"thinking closed {pct(m['think_closed_share'])} | breaker fired {pct(m['breaker_share'])}"]
    if m["grade_errors"]:
        lines.append(f"  WARNING: {m['grade_errors']} grading errors (regrade from texts.jsonl.gz)")
    if m.get("authors"):
        a = m["authors"]
        lines.append(f"  original evaluation on the same {a['problems']} problems: {pct(a['accuracy'])} "
                     f"({a['samples']} samples; first sample only {pct(a['accuracy_first_sample'])})")
    if m.get("gate"):
        g = m["gate"]
        lines.append(f"  GATE: accuracy {pct(m['accuracy'])} in [{pct(g['interval'][0])}, {pct(g['interval'][1])}]? "
                     f"{'PASS' if g['pass'] else 'FAIL -> stop and check the evaluation pipeline'}; "
                     f"cut share {pct(m['cut_share'])} (expected about {pct(g['cut_about'])})")
    return "\n".join(lines)


def summarize(folders, authors_dir=None, gate_model=None, out=None) -> dict:
    """Summary of the folders; prints a report, writes summary.json and per_problem.csv to `out` if given."""
    recs, metas = read_records(folders)
    summary = {"throughput": throughput(metas), "groups": {}}
    for (model, config, bench), rs in sorted(group(recs).items()):
        m = metrics(rs)
        if authors_dir:
            m["authors"] = authors_accuracy(authors_dir, bench, sorted({r["problem_id"] for r in rs}))
        if gate_model and config == "D0" and bench == "math500":
            m["gate"] = gate(gate_model, m)
        key = f"{model}/{config}/{bench}"
        summary["groups"][key] = m
        print("\n" + report(key, m))
    t = summary["throughput"]
    if t["tokens_per_s"]:
        print(f"\nthroughput: {t['tokens']:,} tokens in {t['seconds'] / 3600:.2f} h = {t['tokens_per_s']:,.0f} tok/s")
    if out:
        write(out, summary, recs)
    return summary


def write(out, summary, recs):
    import csv  # noqa: PLC0415

    out = Path(out)
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
