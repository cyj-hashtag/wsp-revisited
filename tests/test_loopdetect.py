import itertools

import pytest

from wsprev.loopdetect import dup_frac, dup_tail, word_dup_tail


def test_distinct_is_zero():
    assert dup_frac(list(range(100)), 8) == 0.0


def test_too_short_is_zero():
    assert dup_frac([1, 2, 3], 8) == 0.0 and dup_frac([], 8) == 0.0


@pytest.mark.parametrize("period, length, n", [(5, 100, 8), (13, 2048, 8), (1, 50, 8), (40, 200, 4)])
def test_periodic(period, length, n):
    seq = list(itertools.islice(itertools.cycle(range(period)), length))
    positions = length - n + 1
    assert dup_frac(seq, n) == pytest.approx(1 - period / positions)


def test_small_example():
    # 3-grams of a b c a b c d: abc bca cab abc bcd -> 5 positions, 4 distinct
    assert dup_frac(list("abcabcd"), 3) == pytest.approx(1 / 5)


def test_tail_window():
    seq = list(range(5000)) + [7, 8] * 1024
    assert dup_tail(seq, 2048, 8) == pytest.approx(1 - 2 / (2048 - 7))
    assert dup_tail(seq, 4096, 8) < dup_tail(seq, 2048, 8)


def test_word_version_is_astras_formula():
    text = "x " * 10 + "Wait, let's check the sum again. " * 400
    words = text.split()[-2500:]
    grams = [tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7))]
    assert word_dup_tail(text) == pytest.approx(1 - len(set(grams)) / max(1, len(grams)))
