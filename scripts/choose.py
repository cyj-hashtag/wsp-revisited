"""The two choices made on practice problems before the RQ1 benchmark runs (prereg/criteria.md sections 3 and 5).

    python scripts/choose.py tau [--write]   # after stage tau: the D3 threshold -> results/tau/choice.json
                                             # --write also sets decoding.D3_TAU in src/wsprev/decoding.py
    python scripts/choose.py config          # after stage select: the base configuration of the cross comparison
                                             # -> results/select/choice.json
Commit what they write before the next stage.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wsprev import data, decoding  # noqa: E402
from wsprev.rq1 import SELECT_CONFIGS, TAU_PROBLEMS, choose_config, choose_tau  # noqa: E402
from wsprev.summary import read_records  # noqa: E402

DECODING = REPO / "src" / "wsprev" / "decoding.py"


def correct_counts(folder: Path, ids: list) -> int:
    """Correct answers in a one-sample run that covers exactly `ids`."""
    recs, _ = read_records([folder])
    got = {r["problem_id"]: r["correct"] for r in recs}
    if sorted(got) != sorted(ids) or len(recs) != len(ids):
        raise SystemExit(f"{folder}: expected one sample for each of {len(ids)} problems, found {len(recs)} records")
    return sum(bool(got[i]) for i in ids)


def set_tau(tau: float, path: Path = DECODING):
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(r"^D3_TAU: float \| None = .*$", f"D3_TAU: float | None = {tau:g}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"could not find the D3_TAU line in {path}")
    path.write_text(new, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["tau", "config"])
    ap.add_argument("--write", action="store_true", help="tau: set decoding.D3_TAU")
    a = ap.parse_args(argv)
    runs = Path(os.environ.get("WSPREV_RUNS", data.home() / "runs"))
    if a.what == "tau":
        ids = [r["id"] for r in data.load("practice:dev")[:TAU_PROBLEMS]]
        counts = {t: correct_counts(runs / "tau" / f"base_D3@{t:g}_practice-dev", ids) for t in decoding.TAU_CANDIDATES}
        choice = choose_tau(counts)
        out = {"rule": "most correct answers on the first 256 practice DEV problems (base, 1 sample); ties: larger tau",
               "problems": len(ids), "correct": {f"{t:g}": c for t, c in counts.items()}, "choice": choice}
        target = REPO / "results" / "tau" / "choice.json"
    else:
        if decoding.D3_TAU is None:
            raise SystemExit("decoding.D3_TAU is not set")
        ids = [r["id"] for r in data.load("practice:select")]
        counts = {c: correct_counts(runs / "select" / f"base_{c}_practice-select", ids) for c in SELECT_CONFIGS}
        choice = choose_config(counts)
        out = {"rule": "most correct answers on the 128 practice SELECT problems (base, 1 sample); ties: D1, D2b, D3",
               "problems": len(ids), "correct": counts, "tau": decoding.D3_TAU, "choice": choice}
        target = REPO / "results" / "select" / "choice.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1), f"\n-> {target}")
    if a.what == "tau" and a.write:
        set_tau(choice)
        print(f"decoding.D3_TAU = {choice:g} written to {DECODING}; commit it with {target}")


if __name__ == "__main__":
    main()
