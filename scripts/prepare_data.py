"""Download the three RQ1 benchmarks into JSONL files with the same fields as the original evaluation
(exp8_math/code/evaluate.py, prepare) and write manifest.json (dataset revisions, row counts, file sha256).

    python scripts/prepare_data.py [--out $WSPREV_BENCH]

Uses HF_ENDPOINT (scripts/env.sh sets the hf-mirror.com mirror).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

SETS = {  # name: (dataset, config, split, expected rows)
    "math500": ("HuggingFaceH4/MATH-500", None, "test", 500),
    "gsm8k": ("openai/gsm8k", "main", "test", 1319),
    "ifeval": ("google/IFEval", None, "train", 541),
}


def rows_of(name, ds):
    if name == "math500":
        return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in ds]
    if name == "gsm8k":
        return [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(ds)]
    return [{"id": f"ifeval_{r['key']}", "kind": "ifeval", "prompt": r["prompt"],
             "instruction_id_list": list(r["instruction_id_list"]), "kwargs": list(r["kwargs"])} for r in ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("WSPREV_BENCH", "/root/autodl-tmp/wsprev/bench"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    from huggingface_hub import HfApi

    manifest = {}
    for name, (repo, config, split, n) in SETS.items():
        try:
            rev = HfApi().dataset_info(repo).sha
        except Exception as e:  # noqa: BLE001 - the mirror may not serve the API; record that
            print(f"{name}: could not read the dataset revision ({e!r}); using the default branch")
            rev = None
        ds = load_dataset(repo, config, split=split, revision=rev)
        rows = rows_of(name, ds)
        assert len(rows) == n, (name, len(rows), n)
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[name] = {"dataset": repo, "config": config, "split": split, "revision": rev, "rows": len(rows),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name, manifest[name])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
