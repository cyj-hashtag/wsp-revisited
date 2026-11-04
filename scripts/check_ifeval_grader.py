"""Our IFEval grader against the original evaluation's: regrade the authors' published first samples.

    python scripts/check_ifeval_grader.py [--out results/ifeval_grader_check.json]

The authors' log repository holds the full text of the first sample of 100 IFEval problems per model
(exp8_math/eval/<unedited|p50c>/parts/*/ifeval_first_samples.jsonl.gz, generated with think35 = D0) and their
per-problem results (ifeval_per_problem.json). The original grader (the private package weightenv) is not public, so
RQ1 grades IFEval with the official google-research checkers (third_party/ifeval, src/wsprev/grade.py). This script
grades the authors' texts with ours and counts the problems where the two disagree. CPU only.
"""
import argparse
import glob
import gzip
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wsprev import data, grade  # noqa: E402

MODELS = {"base": "unedited", "p50c": "p50c"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "results" / "ifeval_grader_check.json"))
    a = ap.parse_args(argv)
    rows = {r["id"]: r for r in data.load("ifeval")}
    root = data.logs_dir() / "exp8_math" / "eval"
    out = {"method": __doc__.split("\n\n", 2)[2].strip(), "models": {}}
    for model, folder in MODELS.items():
        labels, texts = {}, {}
        for p in glob.glob(os.path.join(root, folder, "parts", "*", "ifeval_per_problem.json")):
            with open(p, encoding="utf-8") as f:
                labels.update({r["id"]: r["correct"][0] for r in json.load(f)})
        for p in glob.glob(os.path.join(root, folder, "parts", "*", "ifeval_first_samples.jsonl.gz")):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                texts.update({x["id"]: x["text"] for x in map(json.loads, f)})
        ours = {i: int(grade.grade(rows[i], t, True)["correct"]) for i, t in sorted(texts.items())}
        differ = [{"id": i, "original": labels[i], "ours": ours[i],
                   "instructions": rows[i]["instruction_id_list"]} for i in ours if ours[i] != labels[i]]
        out["models"][model] = {"texts": len(ours), "original_pass": sum(labels[i] for i in ours),
                                "ours_pass": sum(ours.values()), "agree": len(ours) - len(differ), "differ": differ}
        print(f"{model}: {len(ours)} texts, original passes {out['models'][model]['original_pass']}, ours "
              f"{out['models'][model]['ours_pass']}, agree on {len(ours) - len(differ)}")
        for d in differ:
            print(f"  differs: {d}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
