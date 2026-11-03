"""Rank-one column-masked edits of an MLP output matrix (plan section 3, RQ2 and RQ3).

For a down_proj weight W (d_out x d_in; y = W a), a unit vector u (d_out) and a 0/1 column mask m (d_in):

    W' = W - alpha * u (m * (u^T W))          (the form of Astra's p50c edit, alpha = 6, 64 columns per layer)

Proposition 1: W' a = W a - alpha * (c . a) u with c = m * (W^T u). The edit therefore equals adding the
activation-dependent steering term -alpha (c . a) u to the layer's MLP output (SteeringHook). The component of y
orthogonal to u is unchanged; with m = 1, u^T y' = (1 - alpha) u^T y (alpha = 1 projects u out, alpha = 2 reflects,
alpha = 6 flips the sign and scales by 5).

Functions accept numpy arrays or torch tensors (the same type for every argument).
"""
from __future__ import annotations

import numpy as np


def masked_rank1_edit(W, u, mask, alpha: float):
    """W - alpha * u (mask * (u^T W))."""
    r = (u @ W) * mask
    return W - alpha * u[:, None] * r[None, :]


def steering_vector(W, u, mask):
    """c = mask * (W^T u): the edit's output change is -alpha * (c . a) u."""
    return (u @ W) * mask


class SteeringHook:
    """Forward hook for an nn.Linear without bias (mlp.down_proj): output - alpha * (input . c) u.

    Computed in float32 and cast back to the output dtype.  hook = SteeringHook(W, u, mask, alpha);
    handle = hook.attach(model.model.language_model.layers[15].mlp.down_proj); ...; handle.remove()"""

    def __init__(self, W, u, mask, alpha: float):
        import torch  # noqa: PLC0415

        W, u, mask = (torch.as_tensor(x).float() for x in (W, u, mask))
        self.u = u
        self.c = steering_vector(W, u, mask)
        self.alpha = float(alpha)

    def __call__(self, module, inputs, output):  # noqa: ARG002
        a = inputs[0].float()
        s = a @ self.c.to(a.device)
        delta = self.alpha * s.unsqueeze(-1) * self.u.to(a.device)
        return (output.float() - delta).to(output.dtype)

    def attach(self, module):
        return module.register_forward_hook(self)


# ---------------------------------------------------------------- controls (RQ2 step 8)
def random_unit(d: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random unit vector in R^d."""
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def random_mask(d_in: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A 0/1 mask with k ones at uniformly random positions."""
    m = np.zeros(d_in)
    m[rng.choice(d_in, size=k, replace=False)] = 1.0
    return m


def topk_mask(scores, k: int) -> np.ndarray:
    """A 0/1 mask selecting the k largest scores (ties broken by the lower index)."""
    scores = np.asarray(scores, dtype=np.float64)
    m = np.zeros(scores.shape[0])
    m[np.argsort(-scores, kind="stable")[:k]] = 1.0
    return m
