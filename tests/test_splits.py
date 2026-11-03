"""results/splits.json: sizes, no overlap, balanced and interleaved by source."""
import json

from wsprev.data import SPLITS_PATH


def test_splits():
    sp = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    splits = sp["splits"]
    assert {k: len(v) for k, v in splits.items()} == {"dev": 512, "holdout": 256, "select": 128}
    all_ids = [x for v in splits.values() for x in v]
    assert len(set(all_ids)) == len(all_ids)
    for ids in splits.values():
        assert all(x.startswith("gsm8k_train_") for x in ids[0::2])
        assert all(x.startswith("math_train_") for x in ids[1::2])
    assert sp["source"]["commit"] == "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
