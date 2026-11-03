"""Repetition measures for generated text (plan section 3: D3's loop breaker, RQ2's loop windows).

dup_frac(seq, n) is the share of n-gram positions of `seq` whose n-gram already occurred earlier in `seq`,
i.e. 1 - (number of distinct n-grams) / (number of n-gram positions). On token ids it is D3's "dup8".
Astra's statistic in Experiment 8 ("tail duplicate 8-word fraction"; mean 0.958 over the base model's cut practice
answers, exp8_math/runs/p50c/work_notes/final_report.md) is the same formula on the last 2,500 whitespace-separated
words of the text: word_dup_tail.
"""
from __future__ import annotations

import re
from collections.abc import Sequence


def dup_frac(seq: Sequence, n: int = 8) -> float:
    m = len(seq) - n + 1
    if m <= 0:
        return 0.0
    grams = {tuple(seq[i:i + n]) for i in range(m)}
    return 1.0 - len(grams) / m


def dup_tail(ids: Sequence[int], window: int = 2048, n: int = 8) -> float:
    """dup_frac of the last `window` items (D3 checks the last 2,048 thinking tokens)."""
    return dup_frac(ids[-window:] if window else ids, n)


def word_dup_tail(text: str, words: int = 2500, n: int = 8) -> float:
    """Astra's tail statistic: dup_frac of the last `words` whitespace-separated words."""
    return dup_frac(re.findall(r"\S+", text)[-words:], n)
