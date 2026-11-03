"""Summaries of generation runs (output folders of wsprev.generate).

    python scripts/summarize.py <folder> [<folder> ...] [--authors <logs>/exp8_math/eval/unedited] [--gate base|p50c]
                                [--out results/<name>]

A folder holds records.jsonl itself or in sub-folders (shards). Records are grouped by (model, config, benchmark).
Per group: accuracy over all samples (with its standard error over problems), share cut at the token cap, mean
generated tokens, accuracy of finished and of cut answers, share of thinking closed by the model, breaker firing
share and mean position, tokens/s (meta.json).
--authors  the original evaluation's accuracy on the same problems, all of its samples
           (<eval>/parts/*/<benchmark>_per_problem.json).
--gate     the D0 calibration interval for that model (MATH-500); exit status 3 if outside.
--out      writes summary.json and per_problem.csv there (small files that go into git).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsprev.summary import GATES, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--authors", help="the original evaluation folder of the same model (exp8_math/eval/<model>)")
    ap.add_argument("--gate", choices=sorted(GATES))
    ap.add_argument("--out")
    a = ap.parse_args()
    s = summarize(a.folders, a.authors, a.gate, a.out)
    if any(g.get("gate", {}).get("pass") is False for g in s["groups"].values()):
        sys.exit(3)


if __name__ == "__main__":
    main()
