"""Grading, identical to the original Experiment 8 evaluation (exp8_math/code/evaluate.py, run / grade_other).

- MATH-500, GSM8K and the practice problems: the last \\boxed{} answer of the text after the last `</think>` (the whole
  text if there is none), checked against the reference with math-verify 0.9.0 (third_party/wsp_logs/mathqa.py).
- IFEval: strict prompt-level accuracy with the official checkers (third_party/ifeval) on the response after the last
  `</think>`; with thinking on and no `</think>` the response is empty and nothing counts as followed.
"""
from __future__ import annotations

from ._vendor import ifeval_registry, mathqa

PRED_CHARS = 200


def grade(row: dict, text: str, thinking: bool) -> dict:
    if row.get("kind") == "ifeval":
        return grade_ifeval(row, text, thinking)
    return grade_math(row["answer"], text)


def grade_math(gold: str, text: str) -> dict:
    q = mathqa()
    pred, n_boxes = q.extract(text)
    short = pred if pred is None or len(pred) <= PRED_CHARS else pred[:PRED_CHARS] + "..."
    return {"pred": short, "n_boxes": n_boxes, "correct": bool(q.is_correct(pred, gold))}


def ifeval_response(text: str, thinking: bool) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return "" if thinking else text.strip()


def grade_ifeval(row: dict, text: str, thinking: bool) -> dict:
    registry = ifeval_registry()
    resp = ifeval_response(text, thinking)
    follow = []
    for iid, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
        ins = registry.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=row["prompt"])
        follow.append(bool(resp) and bool(ins.check_following(resp)))
    return {"pred": None, "correct": all(follow), "inst_ok": sum(follow), "inst_total": len(follow)}
