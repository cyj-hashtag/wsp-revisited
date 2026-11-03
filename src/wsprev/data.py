"""Benchmarks (prepared by scripts/prepare_data.py) and practice problems (the authors' practice set, split by
scripts/make_splits.py into results/splits.json).

Rows: math benchmarks and practice problems {"id", "problem", "answer", ...}; IFEval {"id", "kind": "ifeval",
"prompt", "instruction_id_list", "kwargs"}.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ._vendor import REPO, mathqa

BENCHMARKS = ("math500", "gsm8k", "ifeval")
RQ1_SAMPLES = {"math500": 2, "gsm8k": 1, "ifeval": 2}   # samples per problem in RQ1 (plan section 3)
BENCH_SIZES = {"math500": 500, "gsm8k": 1319, "ifeval": 541}
SPLITS_PATH = REPO / "results" / "splits.json"
PRACTICE_FILES = ("gsm8k_train.jsonl", "math_train.jsonl")


def home() -> Path:
    return Path(os.environ.get("WSPREV_HOME", "/root/autodl-tmp/wsprev"))


def bench_dir() -> Path:
    return Path(os.environ.get("WSPREV_BENCH", home() / "bench"))


def logs_dir() -> Path:
    return Path(os.environ.get("WSPREV_LOGS", home() / "lm_understand-logs"))


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_lf(path) -> str:
    """sha256 of a text file with LF line endings (a Windows checkout with CRLF gives the same value)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def user_message(row: dict) -> str:
    """The user turn: IFEval's prompt as is; a math problem + "\\n" + Qwen's boxed-answer instruction (mathqa)."""
    if row.get("kind") == "ifeval":
        return row["prompt"]
    return mathqa().build_prompt(row["problem"])


def load(name: str) -> list[dict]:
    """A benchmark ("math500", "gsm8k", "ifeval") or a practice split ("practice:dev", "practice:holdout",
    "practice:select")."""
    if name.startswith("practice:"):
        return load_practice_split(name.split(":", 1)[1])
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}")
    rows = read_jsonl(bench_dir() / f"{name}.jsonl")
    if len(rows) != BENCH_SIZES[name]:
        raise RuntimeError(f"{name}: {len(rows)} rows, expected {BENCH_SIZES[name]}")
    return rows


def load_practice(practice_dir: Path | None = None) -> dict[str, dict]:
    d = Path(practice_dir) if practice_dir else logs_dir() / "exp8_math" / "practice"
    rows = {}
    for f in PRACTICE_FILES:
        for r in read_jsonl(d / f):
            rows[r["id"]] = r
    return rows


def load_practice_split(split: str, splits_path: Path = SPLITS_PATH, practice_dir: Path | None = None) -> list[dict]:
    sp = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    d = Path(practice_dir) if practice_dir else logs_dir() / "exp8_math" / "practice"
    for f, info in sp["source"]["files"].items():
        if sha256_lf(d / f) != info["sha256"]:
            raise RuntimeError(f"{d / f} differs from the file results/splits.json was made from")
    rows = load_practice(d)
    return [rows[i] for i in sp["splits"][split]]
