"""The vendored apply_edit.py rebuilds edited tensors and checks their sha256 (synthetic files here; the real p50c
rebuild, with the authors' hashes, runs in scripts/setup_autodl.sh)."""
import hashlib
import importlib.util
import json

import numpy as np
import pytest

from wsprev._vendor import WSP_LOGS


def load(name):
    spec = importlib.util.spec_from_file_location(f"wsp_logs_{name}", WSP_LOGS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def make(tmp_path, base_sha=None):
    S = load("st_numpy")
    rng = np.random.default_rng(0)
    w = S.bf16_bits(rng.standard_normal((4, 6)).astype(np.float32))
    other = rng.standard_normal(3).astype(np.float32)
    S.save(str(tmp_path / "base.safetensors"), {"w": ("BF16", w), "other": ("F32", other)}, {"format": "pt"})
    idx = np.array([1, 7, 20])
    val = S.bf16_bits(np.array([1.5, -2.0, 0.25], np.float32))
    edited = w.copy().reshape(-1)
    edited[idx] = val
    edited = edited.reshape(w.shape)
    edit = tmp_path / "edit"
    edit.mkdir()
    np.savez(edit / "edit.npz", t0_idx=idx, t0_val=val)
    manifest = {"tensors": [{"name": "w", "shape": [4, 6], "form": "sparse", "changed_entries": 3,
                             "base_sha256": base_sha or sha(w), "edited_sha256": sha(edited)}]}
    (edit / "manifest.json").write_text(json.dumps(manifest))
    return S, edited, other


def test_rebuild(tmp_path):
    S, edited, other = make(tmp_path)
    load("apply_edit").main(str(tmp_path / "base.safetensors"), str(tmp_path / "edit"), str(tmp_path / "out.safetensors"))
    out = S.load(str(tmp_path / "out.safetensors"))
    assert np.array_equal(out["w"][1], edited) and np.array_equal(out["other"][1], other)


def test_wrong_base_is_refused(tmp_path):
    make(tmp_path, base_sha="0" * 64)
    with pytest.raises(AssertionError, match="base tensor differs"):
        load("apply_edit").main(str(tmp_path / "base.safetensors"), str(tmp_path / "edit"),
                                str(tmp_path / "out.safetensors"))
