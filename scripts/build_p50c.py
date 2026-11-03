"""Build the p50c checkpoint from the base model folder and the authors' edit (exp8_math/runs/p50c/edit).

    python scripts/build_p50c.py <base model folder> <lm_understand-logs>/exp8_math/runs/p50c/edit <out folder>

Every edited tensor is rebuilt by the authors' own apply_edit.rebuild and checked against the base and edited sha256 in
the edit's manifest.json, as apply_edit.main does. Unlike apply_edit.main, which loads and rewrites the whole
safetensors file (about 3.5 GB of RAM), this copies the base file and overwrites only the edited tensors' bytes, so it
also runs in AutoDL's no-GPU mode; afterwards it checks that every other byte of the file equals the base file.
The other files of the folder (config, tokenizer, chat template) are copied unchanged.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "wsp_logs"))
import apply_edit  # noqa: E402
import st_numpy as S  # noqa: E402

WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
CHUNK = 1 << 24


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def same_outside(path_a, path_b, ranges):
    """True if the two files have the same size and the same bytes outside `ranges` ([start, end) pairs)."""
    size = os.path.getsize(path_a)
    if size != os.path.getsize(path_b):
        return False
    cuts = [0]
    for start, end in sorted(ranges):
        cuts += [start, end]
    cuts.append(size)
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        for start, end in zip(cuts[0::2], cuts[1::2]):
            fa.seek(start)
            fb.seek(start)
            pos = start
            while pos < end:
                n = min(CHUNK, end - pos)
                if fa.read(n) != fb.read(n):
                    return False
                pos += n
    return True


def main(base_dir, edit_dir, out_dir):
    base_dir, edit_dir, out_dir = Path(base_dir), Path(edit_dir), Path(out_dir)
    man = json.loads((edit_dir / "manifest.json").read_text(encoding="utf-8"))
    z = np.load(edit_dir / "edit.npz")
    head, off = S.read_header(base_dir / WEIGHTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():
        if f.is_file() and f.name not in (WEIGHTS, "wsprev_model.json"):
            shutil.copy2(f, out_dir / f.name)
    tmp = out_dir / (WEIGHTS + ".tmp")
    shutil.copyfile(base_dir / WEIGHTS, tmp)
    ranges = []
    with open(tmp, "r+b") as fo:
        for i, t in enumerate(man["tensors"]):
            info = head[t["name"]]
            assert info["dtype"] == "BF16" and info["shape"] == t["shape"], (t["name"], info)
            start, end = off + info["data_offsets"][0], off + info["data_offsets"][1]
            fo.seek(start)
            base_bits = np.frombuffer(fo.read(end - start), dtype=np.uint16).reshape(t["shape"])
            assert sha(base_bits) == t["base_sha256"], f"{t['name']}: the base tensor differs from the one the edit was made on"
            e = apply_edit.rebuild(base_bits, t, z, i)
            assert sha(e) == t["edited_sha256"], f"{t['name']}: rebuilt tensor does not match the evaluated checkpoint"
            fo.seek(start)
            fo.write(np.ascontiguousarray(e).tobytes())
            ranges.append((start, end))
            print(f"{t['name']}: rebuilt ({t['form']}, {t['changed_entries']:,} entries changed), hash OK")
    with open(tmp, "rb") as fi:     # re-read what was written
        for (start, end), t in zip(ranges, man["tensors"]):
            fi.seek(start)
            assert hashlib.sha256(fi.read(end - start)).hexdigest() == t["edited_sha256"], t["name"]
    assert same_outside(base_dir / WEIGHTS, tmp, ranges), "bytes outside the edited tensors differ from the base file"
    os.replace(tmp, out_dir / WEIGHTS)
    print(f"{len(ranges)} tensors replaced, every other byte equal to the base file -> {out_dir / WEIGHTS}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
