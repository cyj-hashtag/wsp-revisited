"""Grading on known answers (math-verify via the authors' mathqa.py; official IFEval checkers)."""
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from wsprev import data, grade


def run(fn, *args):
    """math-verify's timeouts use signal.alarm on Linux (the GPU machine, main thread) but subprocesses on Windows,
    which fail under pytest; off the main thread mathqa runs the checks without timeouts (mathqa._timeouts)."""
    if sys.platform != "win32":
        return fn(*args)
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(fn, *args).result()


@pytest.mark.parametrize("gold, text, pred, ok", [
    ("\\frac{1}{2}", "<think>so 1/2</think>\n\nThe answer is $\\boxed{\\dfrac{1}{2}}$.", "\\dfrac{1}{2}", True),
    ("72", "<think>maybe \\boxed{70}</think> Final: \\boxed{72}", "72", True),          # after the last </think>
    ("72", "<think>maybe \\boxed{72}</think> Final: \\boxed{70}", "70", False),
    ("5", "thinking that never closes ... \\boxed{5} wait, let me check", "5", True),   # no </think>: whole text
    ("5", "thinking cut at the cap, no answer", None, False),
    ("(3,-1)", "\\boxed{(3, -1)}", "(3, -1)", True),
    ("10", "\\boxed{10\\%}", "10\\%", True),
])
def test_math(gold, text, pred, ok):
    g = run(grade.grade, {"answer": gold}, text, True)
    assert g["pred"] == pred and g["correct"] is ok


def test_user_message_matches_original_prompt():
    msg = data.user_message({"problem": "What is 1+1?"})
    assert msg == "What is 1+1?\nPlease reason step by step, and put your final answer within \\boxed{}."
    assert data.user_message({"kind": "ifeval", "prompt": "Write a haiku."}) == "Write a haiku."


IFEVAL = {"id": "ifeval_1", "kind": "ifeval", "prompt": "Answer without commas, in lowercase.",
          "instruction_id_list": ["punctuation:no_comma", "change_case:english_lowercase"],
          "kwargs": [{"num_highlights": None}, {}]}


@pytest.mark.parametrize("text, thinking, ok, n_ok", [
    ("<think>Hmm, ok.</think>\n\nhello world", True, True, 2),
    ("<think>hmm</think>\n\nHello, world", True, False, 0),
    ("<think>hmm</think>\n\nhello, world", True, False, 1),
    ("hello world, never closed", True, False, 0),       # thinking on, no </think>: empty response
    ("hello world", False, True, 2),                     # thinking off: the whole text
])
def test_ifeval_strict(text, thinking, ok, n_ok):
    pytest.importorskip("langdetect")
    g = grade.grade(IFEVAL, text, thinking)
    assert (g["correct"], g["inst_ok"], g["inst_total"]) == (ok, n_ok, 2)
