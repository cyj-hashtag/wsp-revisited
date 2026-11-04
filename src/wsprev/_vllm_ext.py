"""Raises vLLM 0.33.0's limit of 1,024 logit-bias tokens per request.

D3 carries the presence penalty into the answer as a logit bias on every token id of the kept thinking and of the
closing string (prereg/criteria.md section 3, item 3). A thinking of up to 30,720 tokens can hold more than 1,024
distinct ids (p50c on MATH-500 reached 1,054 on 2026-11-04), which vLLM rejected with "Too many logit bias tokens".
The limit only sizes vLLM's per-request bias buffers and its kernel block; the bias added to the logits is the same.

vLLM imports this module in its worker process as the worker extension class (`worker_extension_cls`, set by
wsprev.generate.VLLMEngine), before the sampler allocates those buffers.
"""
from vllm.v1.worker.gpu.sample import logit_bias as _logit_bias

from . import decoding

# the kept thinking (at most D3_MAX_THINK tokens) plus the 3 closing tokens, rounded up to a power of two
MAX_LOGIT_BIAS_TOKENS = 1 << (decoding.D3_MAX_THINK + 3 - 1).bit_length()
_logit_bias.MAX_NUM_LOGIT_BIAS_TOKENS = max(_logit_bias.MAX_NUM_LOGIT_BIAS_TOKENS, MAX_LOGIT_BIAS_TOKENS)


class LogitBiasLimit:
    """Empty worker extension: importing this module raises the limit."""
