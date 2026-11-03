"""scripts/build_p50c.py (in-place rebuild) and scripts/verify_model.py (hash helpers) on synthetic files."""
import importlib.util
import json

import numpy as np
import pytest
from test_apply_edit import make

from wsprev._vendor import REPO


def script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_p50c_patches_only_the_edited_tensor(tmp_path):
    S, edited, other = make(tmp_path)
    b = script("build_p50c")
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "base.safetensors").rename(base / b.WEIGHTS)
    (base / "config.json").write_text('{"x": 1}')
    b.main(base, tmp_path / "edit", tmp_path / "p50c")
    out = S.load(str(tmp_path / "p50c" / b.WEIGHTS))
    assert np.array_equal(out["w"][1], edited) and np.array_equal(out["other"][1], other)
    assert (tmp_path / "p50c" / "config.json").read_text() == '{"x": 1}'
    assert S.read_header(str(tmp_path / "p50c" / b.WEIGHTS)) == S.read_header(str(base / b.WEIGHTS))


def test_build_p50c_refuses_a_wrong_base(tmp_path):
    make(tmp_path, base_sha="0" * 64)
    b = script("build_p50c")
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "base.safetensors").rename(base / b.WEIGHTS)
    with pytest.raises(AssertionError, match="base tensor differs"):
        b.main(base, tmp_path / "edit", tmp_path / "p50c")
    assert not (tmp_path / "p50c" / b.WEIGHTS).exists()


def test_git_blob_sha1_matches_upstream_blob_id():
    v = script("verify_model")
    # third_party/wsp_logs/README.md lists the upstream git blob of each vendored file
    assert v.git_blob_sha1(REPO / "third_party" / "wsp_logs" / "mathqa.py") == "808a04776906b1cf4630d397cef54f03d8795477"
    assert json.dumps(v.HF_FILES)  # the pinned manifest is well formed


@pytest.mark.parametrize("nx, ny, status", [(135, 1000, "PASS"), (1000, 135, "FAIL"), (500, 500, "WARN")])
def test_bias_order_verdict(nx, ny, status):
    # d = 3, b = -4, T = 0.5: log(p_x/p_y) = -2 if the bias acts before temperature, +2 if after
    assert script("check_vllm").bias_order_verdict(3.0, -4.0, 0.5, nx, ny)[0] == status
