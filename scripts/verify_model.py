"""Check a model folder against Hugging Face's hashes of Qwen/Qwen3.5-0.8B at the pinned revision and write
wsprev_model.json (name, source, revision, sha256 of the weight file) into the folder; wsprev.generate reads it.

    python scripts/verify_model.py base <folder>
    python scripts/verify_model.py p50c <folder> --edit <lm_understand-logs>/exp8_math/runs/p50c/edit

For p50c the weight file is not the release's (scripts/build_p50c.py checks its tensors); its sha256 is recorded.
Exit status 1 if a file is missing or differs.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
# https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B/tree/<REVISION>, fetched 2026-11-03:
# ("sha256", lfs oid) for files stored in LFS, ("blob", git blob sha1) for the others.
HF_FILES = {
    WEIGHTS: ("sha256", "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"),
    "tokenizer.json": ("sha256", "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    "chat_template.jinja": ("blob", "0ef09f214eaa6d9bca297988afc1454b5827b2c7"),
    "config.json": ("blob", "715f0448b9d38103211f0ad88bbb4d6e4f4be8c9"),
    "merges.txt": ("blob", "a494e019ca1502219fd0128658b979e5f05ae8e8"),
    "model.safetensors.index.json": ("blob", "f691cefdb79d73270895ebd6d9594ddcecfc1838"),
    "preprocessor_config.json": ("blob", "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
    "tokenizer_config.json": ("blob", "fae3ce993e07c092ad024dde45e592379fde91bb"),
    "video_preprocessor_config.json": ("blob", "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe"),
    "vocab.json": ("blob", "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
    "LICENSE": ("blob", "f938136e3adacfd92be087f6e113b5d6d97f678f"),
    "README.md": ("blob", "5824f1761b2b3a55a2141a9a1172a7f92c7c2ad9"),
}
OPTIONAL = {"LICENSE", "README.md"}


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["base", "p50c"])
    ap.add_argument("folder")
    ap.add_argument("--edit", help="the p50c edit folder (manifest.json, edit.npz)")
    a = ap.parse_args(argv)
    folder = Path(a.folder)
    bad = []
    for fname, (kind, want) in HF_FILES.items():
        path = folder / fname
        if a.name == "p50c" and fname == WEIGHTS:
            continue
        if not path.exists():
            if fname not in OPTIONAL:
                bad.append(f"missing {fname}")
            continue
        got = file_sha256(path) if kind == "sha256" else git_blob_sha1(path)
        print(f"{'ok ' if got == want else 'BAD'} {fname}")
        if got != want:
            bad.append(f"{fname}: {got} != {want}")
    if bad:
        print("\n".join(bad))
        sys.exit(1)
    info = {"name": a.name, "repo": REPO_ID, "revision": REVISION, "weight_file": WEIGHTS,
            "weight_sha256": file_sha256(folder / WEIGHTS), "edit": None}
    if a.name == "p50c":
        if not a.edit:
            sys.exit("p50c needs --edit")
        man = Path(a.edit) / "manifest.json"
        info["edit"] = {"manifest": str(man), "manifest_sha256": file_sha256(man),
                        "tensors": [t["name"] for t in json.loads(man.read_text(encoding="utf-8"))["tensors"]],
                        "built_by": "scripts/build_p50c.py (authors' apply_edit.rebuild, sha256-checked)"}
    (folder / "wsprev_model.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"{a.name}: all files match; weight sha256 {info['weight_sha256']} -> {folder / 'wsprev_model.json'}")


if __name__ == "__main__":
    main()
