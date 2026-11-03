"""Proposition 1: the column-masked rank-one weight edit equals an activation-dependent steering hook."""
import numpy as np
import pytest

from wsprev.edits import SteeringHook, masked_rank1_edit, random_mask, random_unit, steering_vector, topk_mask

D_OUT, D_IN = 32, 96


def setup(seed=0, k=12):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((D_OUT, D_IN))
    u = random_unit(D_OUT, rng)
    m = random_mask(D_IN, k, rng)
    a = rng.standard_normal((5, D_IN))
    return W, u, m, a


@pytest.mark.parametrize("alpha", [1.0, 2.0, 6.0, -0.5])
def test_prop1_weight_equals_activation_form(alpha):
    W, u, m, a = setup()
    Wp = masked_rank1_edit(W, u, m, alpha)
    c = steering_vector(W, u, m)
    np.testing.assert_allclose(a @ Wp.T, a @ W.T - alpha * np.outer(a @ c, u), atol=1e-10)


def test_orthogonal_part_unchanged():
    W, u, m, a = setup()
    P = np.eye(D_OUT) - np.outer(u, u)
    np.testing.assert_allclose(a @ masked_rank1_edit(W, u, m, 6.0).T @ P, a @ W.T @ P, atol=1e-10)


@pytest.mark.parametrize("alpha, factor", [(1.0, 0.0), (2.0, -1.0), (6.0, -5.0)])
def test_full_mask_scales_u_component(alpha, factor):
    W, u, _, a = setup()
    y, yp = a @ W.T @ u, a @ masked_rank1_edit(W, u, np.ones(D_IN), alpha).T @ u
    np.testing.assert_allclose(yp, factor * y, atol=1e-10)


def test_only_masked_columns_change():
    W, u, m, _ = setup()
    changed = np.any(masked_rank1_edit(W, u, m, 6.0) != W, axis=0)
    assert np.array_equal(changed, m.astype(bool))


def test_masks():
    rng = np.random.default_rng(1)
    assert random_mask(100, 64, rng).sum() == 64
    assert np.array_equal(np.flatnonzero(topk_mask([0.1, 5, 3, 5, -1], 2)), [1, 3])


def test_hook_equals_weight_edit_in_torch():
    torch = pytest.importorskip("torch")
    W, u, m, a = setup(seed=3)
    for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
        lin = torch.nn.Linear(D_IN, D_OUT, bias=False)
        lin.weight.data = torch.tensor(W, dtype=torch.float32)
        edited = torch.nn.Linear(D_IN, D_OUT, bias=False)
        edited.weight.data = torch.tensor(masked_rank1_edit(W, u, m, 6.0), dtype=torch.float32)
        lin, edited = lin.to(dtype), edited.to(dtype)
        x = torch.tensor(a, dtype=dtype)
        handle = SteeringHook(W, u, m, 6.0).attach(lin)
        with torch.no_grad():
            hooked, ref = lin(x).float(), edited(x).float()
        handle.remove()
        scale = ref.abs().max().item()
        assert (hooked - ref).abs().max().item() <= tol * scale
