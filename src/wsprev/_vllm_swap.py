"""vLLM worker extension for RQ2's screening: replace MLP down_proj weights of the loaded model in place, and restore
them, so that one engine evaluates many edits without reloading the model (wsprev.rq2).

The new values are copied into the existing parameter tensors (same storage), as vLLM's own weight-update examples
do, so captured CUDA graphs see them. wsprev.rq2 resets the prefix cache after every change, and its swap check
compares an edit applied this way with the same edit loaded from disk before any screening run.

It also raises the logit-bias limit (wsprev._vllm_ext), so D3 runs on it as well. Used as
worker_extension_cls="wsprev._vllm_swap.WeightSwap"; the methods are called through LLM.collective_rpc.
"""
from __future__ import annotations

from ._vllm_ext import LogitBiasLimit
from .directions import find_down_proj


def _model(worker):
    runner = worker.model_runner
    model = runner.get_model() if hasattr(runner, "get_model") else runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model   # a CUDA-graph wrapper in some vLLM versions


class WeightSwap(LogitBiasLimit):
    """Methods run in vLLM's worker process (self is the worker)."""

    def wsprev_set_down_proj(self, weights: dict) -> list:
        """weights: {layer: bfloat16 CPU tensor of the parameter's shape}. Keeps the first original of every touched
        parameter for wsprev_restore. Returns the parameter names."""
        import torch  # noqa: PLC0415

        model = _model(self)
        if not hasattr(self, "_wsprev_orig"):
            self._wsprev_orig = {}
        params = list(model.named_parameters())
        names = []
        with torch.no_grad():
            for layer, w in sorted(weights.items()):
                name, p = find_down_proj(params, int(layer))
                if tuple(p.shape) != tuple(w.shape):
                    raise RuntimeError(f"{name}: shape {tuple(p.shape)} in the engine, {tuple(w.shape)} given")
                self._wsprev_orig.setdefault(name, p.detach().clone())
                p.data.copy_(w.to(device=p.device, dtype=p.dtype))
                names.append(name)
        torch.cuda.synchronize()
        return names

    def wsprev_restore(self) -> list:
        """Copy every touched parameter back to its original. Returns the parameter names."""
        import torch  # noqa: PLC0415

        orig = getattr(self, "_wsprev_orig", {})
        params = dict(_model(self).named_parameters())
        with torch.no_grad():
            for name, w in orig.items():
                params[name].data.copy_(w)
        torch.cuda.synchronize()
        return sorted(orig)

    def wsprev_get_down_proj(self, layer: int):
        """The current layer-`layer` down_proj weight as a CPU tensor (for checks)."""
        _, p = find_down_proj(list(_model(self).named_parameters()), int(layer))
        return p.detach().to("cpu").clone()
