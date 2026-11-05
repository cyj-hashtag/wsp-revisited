"""RQ2: loop directions and neuron scores from teacher-forced activations, and the grid of edits
(docs/plan-for-agents.md section 3, RQ2; the fixed rules are in prereg/rq2_rules.md). No LLM is called anywhere.

Steps (the GPU parts run through wsprev.rq2):
1. windows(): from the base model's D0 generations on practice DEV problems, pairs of 2,048-token windows of the
   same response: the LOOP window is the last 2,048 thinking tokens of a response cut at the token limit, used if its
   dup8 >= 0.8; the NORMAL window is the first 2,048 thinking tokens of the same response, used if its dup8 <= 0.3.
   Controls: the middle 2,048 thinking tokens of responses that finished (FINISHED windows).
2. Reader: teacher forcing with HF transformers (forward hooks, no gradients). For every window and every layer l in
   LAYERS it keeps the mean over the window's positions of (i) the residual stream at the output of layer l and
   (ii) the input a of that layer's mlp.down_proj. The context is the prompt plus the thinking up to the window's end.
3. directions(): u_l = (mean over loop windows - mean over normal windows) of the layer-l residual, normalized;
   the layers' separability is the cross-fitted AUC of the projections (u from one half of the pairs, AUC on the
   other half, and the reverse; averaged).
4. scores(): c_j = (u_l^T w_j) (E_loop[a_j] - E_normal[a_j]), w_j the j-th column of W_l (Proposition 3).
5. grid(): edits W' = W - alpha u (m_k * (u^T W)) per layer of a layer set, m_k the top-k columns by c_j.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .edits import masked_rank1_edit, topk_mask
from .loopdetect import dup_frac

LAYERS = tuple(range(8, 23))            # scanned layers l = 8..22 (0-indexed decoder layers)
ASTRA_LAYERS = (15, 19)                 # the layers of p50c's edit
WINDOW = 2048
LOOP_DUP, NORMAL_DUP = 0.8, 0.3
KS = (32, 64, 128, 3584)
ALPHAS = (1, 2, 4, 6, 8)
THINK_END = "</think>"
TENSOR = "model.language_model.layers.{}.mlp.down_proj.weight"   # name in the safetensors file


# ---------------------------------------------------------------- 1. windows
@dataclass
class Window:
    problem_id: str
    kind: str              # "loop", "normal" or "finished"
    ids: list              # the context: prompt + thinking up to the window's end
    start: int             # the window is ids[start:]
    dup8: float
    pair: int | None = None   # index of the loop/normal pair


def thinking(text: str) -> str:
    """The thinking part of a generated text: everything before the model's first `</think>`."""
    i = text.find(THINK_END)
    return text if i < 0 else text[:i]


def windows(records, texts: dict, encode, prompt_ids, window: int = WINDOW, loop_dup: float = LOOP_DUP,
            normal_dup: float = NORMAL_DUP, max_pairs: int | None = None) -> tuple[list[Window], dict]:
    """Windows of the records in the given order (the order of the problems in the split).

    texts: {(problem_id, sample_idx): text}; encode(text) -> token ids (no special tokens); prompt_ids(problem_id) ->
    the prompt's token ids. Returns (windows, counts of why responses were skipped)."""
    out, why, pairs = [], {"cut": 0, "finished": 0, "loop dup8 < min": 0, "normal dup8 > max": 0,
                           "thinking too short": 0, "other": 0}, 0
    for r in records:
        text = texts[(r["problem_id"], r["sample_idx"])]
        ids = list(encode(thinking(text)))
        cut_open = r["cut"] and r["think_closed"] is False and not r.get("loop_triggered")
        finished = (not r["cut"]) and r["think_closed"] and r["finish_reason"] == "stop"
        if cut_open:
            why["cut"] += 1
            if max_pairs is not None and pairs >= max_pairs:
                continue
            if len(ids) < 2 * window:
                why["thinking too short"] += 1
                continue
            d_loop, d_norm = dup_frac(ids[-window:], 8), dup_frac(ids[:window], 8)
            if d_loop < loop_dup:
                why["loop dup8 < min"] += 1
                continue
            if d_norm > normal_dup:
                why["normal dup8 > max"] += 1
                continue
            p = list(prompt_ids(r["problem_id"]))
            out.append(Window(r["problem_id"], "loop", p + ids, len(p) + len(ids) - window, d_loop, pairs))
            out.append(Window(r["problem_id"], "normal", p + ids[:window], len(p), d_norm, pairs))
            pairs += 1
        elif finished:
            why["finished"] += 1
            if len(ids) < window:
                why["thinking too short"] += 1
                continue
            mid = len(ids) // 2
            lo = max(0, mid - window // 2)
            p = list(prompt_ids(r["problem_id"]))
            out.append(Window(r["problem_id"], "finished", p + ids[:lo + window], len(p) + lo,
                              dup_frac(ids[lo:lo + window], 8)))
        else:
            why["other"] += 1
    return out, why


# ---------------------------------------------------------------- 2. teacher-forced reading
class _Stop(Exception):
    pass


def text_model(model):
    """The decoder stack of an HF model: Qwen3_5Model.language_model, ...ForCausalLM.model, or the model itself."""
    for attr in ("language_model", "model"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "layers"):
            return sub
    if hasattr(model, "layers"):
        return model
    for attr in ("model", "language_model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            return text_model(sub)
    raise ValueError("no decoder layers found")


@dataclass
class Reader:
    """Means of the layer outputs and of the down_proj inputs over a window's positions (float32)."""
    model: object
    layers: tuple = LAYERS
    _start: int = 0
    _acc: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tm = text_model(self.model)
        self.handles = []
        last = max(self.layers)
        for l in self.layers:
            layer = self.tm.layers[l]
            self.handles.append(layer.mlp.down_proj.register_forward_hook(self._hook_act(l)))
            self.handles.append(layer.register_forward_hook(self._hook_out(l, l == last)))

    def _mean(self, x):
        return x[0, self._start:].float().mean(dim=0).cpu().numpy()

    def _hook_act(self, l):
        def hook(module, inputs, output):  # noqa: ARG001
            self._acc[("act", l)] = self._mean(inputs[0])
        return hook

    def _hook_out(self, l, stop):
        def hook(module, inputs, output):  # noqa: ARG001
            h = output[0] if isinstance(output, tuple) else output
            self._acc[("res", l)] = self._mean(h)
            if stop:
                raise _Stop
        return hook

    def read(self, ids, start: int) -> tuple[np.ndarray, np.ndarray]:
        """(residual means: len(layers) x d_model, down_proj input means: len(layers) x d_ff) of ids[start:]."""
        import torch  # noqa: PLC0415

        self._start, self._acc = start, {}
        dev = next(self.tm.parameters()).device
        with torch.no_grad():
            try:
                self.tm(input_ids=torch.tensor([list(ids)], device=dev), use_cache=False)
            except _Stop:
                pass
        res = np.stack([self._acc[("res", l)] for l in self.layers])
        act = np.stack([self._acc[("act", l)] for l in self.layers])
        return res, act

    def close(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------- 3. directions and their separability
def auc(pos, neg) -> float:
    """P(X > Y) + P(X = Y) / 2 for X from pos, Y from neg (the Mann-Whitney AUC)."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero mean difference")
    return v / n


def directions(res_loop, res_norm) -> dict:
    """res_loop, res_norm: (pairs, layers, d) window means, pair i of both from the same response.

    Returns u (layers x d, from every pair), and per layer the cross-fitted AUC and standardized separation d' of the
    projections (folds: even and odd pair indices)."""
    res_loop, res_norm = np.asarray(res_loop, dtype=np.float64), np.asarray(res_norm, dtype=np.float64)
    n, L, _ = res_loop.shape
    if n < 4:
        raise ValueError(f"{n} pairs; need at least 4")
    u = np.stack([unit(res_loop[:, l].mean(0) - res_norm[:, l].mean(0)) for l in range(L)])
    folds = (np.arange(n) % 2 == 0, np.arange(n) % 2 == 1)
    aucs, seps = np.zeros(L), np.zeros(L)
    for l in range(L):
        a_, s_ = [], []
        for fit, ev in (folds, folds[::-1]):
            v = unit(res_loop[fit, l].mean(0) - res_norm[fit, l].mean(0))
            x, y = res_loop[ev, l] @ v, res_norm[ev, l] @ v
            a_.append(auc(x, y))
            sd = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) if len(x) > 1 else 0.0
            s_.append(float((x.mean() - y.mean()) / sd) if sd > 0 else math.inf)
        aucs[l], seps[l] = np.mean(a_), np.mean(s_)
    return {"u": u, "auc": aucs, "separation": seps}


def top_layers(layers, aucs, seps, n: int = 2) -> tuple:
    """The n layers with the highest AUC; ties by the larger separation, then by the lower layer index."""
    order = sorted(range(len(layers)), key=lambda i: (-aucs[i], -seps[i], layers[i]))
    return tuple(sorted(layers[i] for i in order[:n]))


# ---------------------------------------------------------------- 4. neuron scores
def scores(W, u, act_loop_mean, act_norm_mean) -> np.ndarray:
    """c_j = (u^T w_j) (E_loop[a_j] - E_normal[a_j]) for W (d_model x d_ff), u (d_model)."""
    W = np.asarray(W, dtype=np.float64)
    return (np.asarray(u, dtype=np.float64) @ W) * (np.asarray(act_loop_mean) - np.asarray(act_norm_mean))


def projection_gap_change(W, u, mask, alpha, act_loop_mean, act_norm_mean) -> dict:
    """First-order check of Proposition 3: the edit changes the loop-minus-normal gap of the mean projection on u at
    the layer's output by -alpha * sum_{j in S} c_j (exact for the mean MLP output, as the MLP input is unchanged)."""
    c = scores(W, u, act_loop_mean, act_norm_mean)
    d = np.asarray(act_loop_mean, dtype=np.float64) - np.asarray(act_norm_mean, dtype=np.float64)
    Wp = masked_rank1_edit(np.asarray(W, dtype=np.float64), np.asarray(u, dtype=np.float64), mask, alpha)
    before, after = float(u @ (W @ d)), float(u @ (Wp @ d))
    return {"before": before, "after": after, "predicted_change": float(-alpha * (c * mask).sum()),
            "change": after - before}


# ---------------------------------------------------------------- 5. the grid of edits
@dataclass(frozen=True)
class Edit:
    name: str
    layer_set: str          # "astra" or "auc"
    layers: tuple
    k: int
    alpha: float

    def to_dict(self):
        return {"name": self.name, "layer_set": self.layer_set, "layers": list(self.layers), "k": self.k,
                "alpha": self.alpha}


def grid(auc_layers, ks=KS, alphas=ALPHAS, astra_layers=ASTRA_LAYERS) -> list[Edit]:
    """Layer sets x k x alpha, in this order (the order breaks ties, prereg/rq2_rules.md). If the AUC layers are
    Astra's, the grid has one layer set."""
    sets = [("astra", tuple(astra_layers))]
    if tuple(sorted(auc_layers)) != tuple(sorted(astra_layers)):
        sets.append(("auc", tuple(sorted(auc_layers))))
    return [Edit(f"{s}-L{'-'.join(map(str, ls))}-k{k}-a{a:g}", s, ls, k, float(a))
            for s, ls in sets for k in ks for a in alphas]


def masks(c_by_layer: dict, k: int) -> dict:
    """{layer: 0/1 mask of the top-k columns by c} (k = d_ff: every column)."""
    return {l: topk_mask(c, k) for l, c in c_by_layer.items()}


def edited(W_by_layer: dict, u_by_layer: dict, mask_by_layer: dict, alpha: float) -> dict:
    """{layer: W'} in float32 for the layers of the edit (computed in float64)."""
    return {l: masked_rank1_edit(np.asarray(W_by_layer[l], dtype=np.float64), np.asarray(u_by_layer[l]),
                                 mask_by_layer[l], alpha).astype(np.float32) for l in mask_by_layer}


def find_down_proj(named_parameters, layer: int) -> tuple[str, object]:
    """The language model's layer-`layer` down_proj weight among (name, parameter) pairs, not the vision tower's
    (used in vLLM's worker by wsprev._vllm_swap)."""
    suffix = f"layers.{layer}.mlp.down_proj.weight"
    hits = [(n, p) for n, p in named_parameters
            if (n == suffix or n.endswith("." + suffix)) and "visual" not in n and "vision" not in n]
    if len(hits) != 1:
        raise RuntimeError(f"expected one parameter ending in {suffix!r}, found {[n for n, _ in hits]}")
    return hits[0]


def recover_rank1(W0, W1, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """(u, mask) of an edit W1 = W0 - alpha u (m * (u^T W0)) stored only as its result (p50c's edit is stored as the
    changed bfloat16 entries): m = the columns that changed; u = the leading left singular vector of W1 - W0 on them.
    Its sign is arbitrary: u and -u give the same edit. `alpha` is not needed for u and m; it is kept in the signature
    so that callers state the edit they assume (rebuilding W1 from u, m and alpha is their check)."""
    D = np.asarray(W1, dtype=np.float64) - np.asarray(W0, dtype=np.float64)
    m = (np.abs(D).sum(axis=0) > 0).astype(float)
    U, _, _ = np.linalg.svd(D[:, m > 0], full_matrices=False)
    u = U[:, 0]
    return u / np.linalg.norm(u), m
