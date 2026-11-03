"""Decoding configurations D0-D4 of RQ1 (docs/plan-for-agents.md, section 3).

This module is the only place where they are defined; scripts refer to them by name: "D0", "D1", "D2a", "D2b", "D3",
"D4", and "D3@<tau>" for D3 with an explicit threshold (the calibration runs on practice problems).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

MAX_TOKENS = 32768               # new tokens per answer (thinking + answer) in every configuration, as in think35
MAX_MODEL_LEN = MAX_TOKENS + 2048
SEED = 0
CLOSE_THINK = "\n</think>\n\n"   # appended to end the thinking from outside (D3, D4); see the model card

# D3's threshold on the duplicate-8-gram share. It is chosen among TAU_CANDIDATES on the practice DEV split by the
# procedure in prereg/criteria.md, before any benchmark run of D3; until then "D3" refuses to run ("D3@<tau>" runs).
TAU_CANDIDATES = (0.3, 0.5, 0.7)
D3_TAU: float | None = None
D3_MAX_THINK = 30720


@dataclass(frozen=True)
class Breaker:
    """Ends the thinking from outside: D3's loop breaker and D4's budget forcing.

    The thinking is one uninterrupted generation with the configuration's sampling, the same request as without a
    breaker. When it reaches t thinking tokens, for every multiple t of `every` before `</think>`, the breaker fires if
      - `tau` is set and the duplicate-n-gram share (loopdetect.dup_frac, n = `ngram`) of the last `window` thinking
        tokens is >= tau, or
      - t >= `max_think`.
    It then cuts the thinking at t, appends CLOSE_THINK and samples at most `answer_tokens` new tokens as the answer
    with the same sampling.

    carry_presence: vLLM's presence penalty only counts a request's own output tokens, so a new answer request would
    forget the thinking. With carry_presence, every token id of the kept thinking and of CLOSE_THINK gets the logit bias
    -presence_penalty and the answer request's own presence penalty is 0. This equals the penalty of one uninterrupted
    generation except for tokens that occur in the answer but nowhere in the thinking (not penalized when repeated).
    """
    tau: float | None
    max_think: int
    every: int = 1024
    window: int = 2048
    ngram: int = 8
    answer_tokens: int = 2048
    carry_presence: bool = True


@dataclass(frozen=True)
class Decoding:
    name: str
    thinking: bool           # chat template enable_thinking
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = MAX_TOKENS
    seed: int = SEED
    breaker: Breaker | None = None
    note: str = ""

    def sampling(self, **override) -> dict:
        """Keyword arguments for vllm.SamplingParams (the caller adds n, seed and output_kind)."""
        sp = {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
              "presence_penalty": self.presence_penalty, "min_p": self.min_p,
              "repetition_penalty": self.repetition_penalty, "max_tokens": self.max_tokens}
        sp.update(override)
        return sp

    def sample_seed(self, k: int) -> int:
        """Seed of the k-th sample of a problem. vLLM V1 splits a request with n > 1 into child requests with seeds
        seed + index (TODO(verify) in vllm/v1/engine/parallel_sampling.py on the GPU machine); we send each sample as
        its own request with that seed."""
        return self.seed + k

    def to_dict(self) -> dict:
        return asdict(self)


D0 = Decoding("D0", True, 0.6, 0.95, 20, 1.5, note="the original think35 sampling of Experiment 8")
D1 = Decoding("D1", True, 1.0, 0.95, 20, 1.5, note="model card: thinking, text tasks; its language benchmarks")
D2a = Decoding("D2a", False, 1.0, 1.00, 20, 2.0, note="model card: non-thinking, text tasks")
D2b = Decoding("D2b", False, 0.7, 0.80, 20, 1.5,
               note="model card: non-thinking benchmark setting (the authors' nothink35 sampling, here with 32,768 tokens)")
D4 = replace(D0, name="D4", breaker=Breaker(tau=None, max_think=8192),
             note="D0 + a thinking budget of 8,192 tokens (s1-style budget forcing)")

_FIXED = {c.name: c for c in (D0, D1, D2a, D2b, D4)}


def d3(tau: float) -> Decoding:
    """D0 + the loop breaker with threshold tau."""
    return replace(D0, name=f"D3@{tau:g}", breaker=Breaker(tau=tau, max_think=D3_MAX_THINK),
                   note="D0 + loop breaker")


def get(name: str) -> Decoding:
    if name.startswith("D3@"):
        return d3(float(name[3:]))
    if name == "D3":
        if D3_TAU is None:
            raise ValueError("D3's tau has not been calibrated yet (decoding.D3_TAU is None); "
                             "calibration runs use D3@<tau>, see prereg/criteria.md")
        return replace(d3(D3_TAU), name="D3")
    if name in _FIXED:
        return _FIXED[name]
    raise KeyError(f"unknown decoding configuration {name!r}; known: {sorted(_FIXED)} + D3, D3@<tau>")
