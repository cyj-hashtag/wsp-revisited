"""Split the authors' practice problems into DEV / HOLDOUT / SELECT (plan section 3: RQ1 configuration choice and D3's
threshold, RQ2's search) and write results/splits.json. Run once; the file is committed before any run uses it.

    python scripts/make_splits.py <lm_understand-logs>/exp8_math/practice

Each split is half GSM8K train, half MATH train (uniformly at random within each source, seed SEED), interleaved
GSM8K, MATH, GSM8K, ... so that every prefix of even length is balanced. The three splits do not overlap.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsprev.data import PRACTICE_FILES, SPLITS_PATH, read_jsonl, sha256_lf  # noqa: E402

SEED = 20261103
SIZES = {"dev": 512, "holdout": 256, "select": 128}
UPSTREAM = {"repo": "https://github.com/RikiyaT/lm_understand-logs",
            "commit": "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f", "path": "exp8_math/practice"}


def main(practice_dir):
    d = Path(practice_dir)
    rng = np.random.default_rng(SEED)
    files, picked = {}, {}
    for f in PRACTICE_FILES:
        ids = [r["id"] for r in read_jsonl(d / f)]
        files[f] = {"sha256": sha256_lf(d / f), "n": len(ids)}
        picked[f] = [ids[i] for i in rng.permutation(len(ids))[:sum(SIZES.values()) // 2]]
    splits, start = {}, 0
    for name, size in SIZES.items():
        g, m = (picked[f][start:start + size // 2] for f in PRACTICE_FILES)
        splits[name] = [x for pair in zip(g, m) for x in pair]
        start += size // 2
    assert len({x for s in splits.values() for x in s}) == sum(SIZES.values())
    out = {"seed": SEED, "sizes": SIZES,
           "method": "per source (gsm8k_train, math_train): numpy default_rng(seed).permutation of the file order, "
                     "first 448 ids; dev = 0-255, holdout = 256-383, select = 384-447 of each source; "
                     "each split interleaves gsm8k, math, gsm8k, ...",
           "source": dict(UPSTREAM, files=files), "splits": splits}
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in splits.items()}, "->", SPLITS_PATH)


if __name__ == "__main__":
    main(sys.argv[1])
