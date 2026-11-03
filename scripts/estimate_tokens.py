"""Tokens each pre-registered run will generate, estimated before any generation from the original evaluation's texts.

    python scripts/estimate_tokens.py <tokenizer.json> [--logs $WSPREV_LOGS] [--out results/cost_estimate.json]

The authors' logs hold the full text of the first sample of 200 MATH-500, 200 GSM8K and 100 IFEval problems per model
(exp8_math/eval/<unedited|p50c>/parts/*/<bench>_first_samples.jsonl.gz), generated with think35 = D0. Each text is
re-tokenized with the model's tokenizer, which gives D0's length, and the registered D3 breaker (wsprev.decoding,
wsprev.loopdetect) is replayed on its thinking tokens: D3 shares D0's thinking until the breaker fires, so this also
gives D3's length for each tau. Only lengths and breaker positions are read, never correctness, so nothing here can
inform a choice that the pre-registration fixes on practice problems.

Assumptions (not measurable from the logs): D1 generates as many tokens as D0; the thinking-off configurations D2a and
D2b generate ASSUMED_NO_THINK tokens per sample; an answer after a breaker cut has ANSWER tokens (low: a guess; high:
the 2,048-token cap, used for the totals).
"""
import argparse
import glob
import gzip
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wsprev import data, decoding  # noqa: E402
from wsprev.loopdetect import dup_frac  # noqa: E402

MODELS = {"base": "unedited", "p50c": "p50c"}            # our name: folder of the original evaluation
BENCHES = ("math500", "gsm8k", "ifeval")
ANSWER = {"low": 300, "high": 2048}
ASSUMED_NO_THINK = {"math500": 2500, "gsm8k": 800, "ifeval": 800}
PRACTICE_MIX = ("math500", "gsm8k")                      # practice splits are half MATH train, half GSM8K train


def checkpoints(ids, close, br):
    """(t, dup) at every checkpoint the breaker of generate.Runner evaluates: t = every, 2 every, ... while the
    thinking has t tokens before </think> (index `close`, or None) and the request has not ended in that step."""
    think = close if close is not None else len(ids)
    seen, t = [], br.every
    while t <= think and (close is not None or t < len(ids)):
        seen.append((t, dup_frac(ids[max(0, t - br.window):t], br.ngram)))
        if t >= br.max_think:
            break
        t += br.every
    return seen


def first_fire(seen, tau, max_think):
    return next((t for t, d in seen if (tau is not None and d >= tau) or t >= max_think), None)


def per_sample(texts, tok):
    """Mean tokens per sample for D0 and for D3 at each tau, from the original evaluation's texts."""
    think_end = tok.token_to_id("</think>")
    n_close = len(tok.encode(decoding.CLOSE_THINK, add_special_tokens=False).ids)
    br = decoding.d3(decoding.TAU_CANDIDATES[0]).breaker
    rows = []
    for enc in tok.encode_batch(texts, add_special_tokens=False):
        ids = enc.ids
        close = ids.index(think_end) if think_end in ids else None
        rows.append((len(ids), close, checkpoints(ids, close, br)))
    n = len(rows)
    out = {"texts": n, "D0": sum(r[0] for r in rows) / n, "closed": sum(r[1] is not None for r in rows) / n}
    for tau in decoding.TAU_CANDIDATES:
        fires = [first_fire(r[2], tau, br.max_think) for r in rows]
        out[f"D3@{tau:g}"] = {
            "fired": sum(f is not None for f in fires) / n,
            **{f"tokens_{k}": sum(f + n_close + a if f is not None else r[0] for f, r in zip(fires, rows)) / n
               for k, a in ANSWER.items()}}
    return out


def stage_totals(m):
    """Tokens of each stage of scripts/run_stage.py under the registered design (D3 at the largest tau, answers at
    the 2,048-token cap)."""
    tau_hi = f"D3@{max(decoding.TAU_CANDIDATES):g}"

    def mean(model, bench, cfg):
        if cfg in ("D2a", "D2b"):
            return ASSUMED_NO_THINK[bench]
        cell = m[f"{model}/{bench}"]
        return cell["D0"] if cfg in ("D0", "D1") else cell[cfg]["tokens_high"]

    def practice(model, cfg):
        return sum(mean(model, b, cfg) for b in PRACTICE_MIX) / len(PRACTICE_MIX)

    totals = {
        "calib (D0, MATH-500, 1 sample)": sum(500 * mean(model, "math500", "D0") for model in MODELS),
        "tau (base, 256 DEV, 3 taus)": 256 * sum(practice("base", f"D3@{t:g}") for t in decoding.TAU_CANDIDATES),
        "select (base, 128 SELECT)": 128 * sum(practice("base", c) for c in ("D1", "D2b", tau_hi)),
    }
    for bench in BENCHES:
        n = data.BENCH_SIZES[bench] * data.RQ1_SAMPLES[bench]
        for cfg in ("D1", "D2a", "D2b", "D3"):
            totals[f"rq1 {bench} {cfg}"] = sum(
                n * mean(model, bench, tau_hi if cfg == "D3" else cfg) for model in MODELS)
    totals["all"] = sum(totals.values())
    return totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizer", help="tokenizer.json of Qwen/Qwen3.5-0.8B (the base model folder has it)")
    ap.add_argument("--logs", default=os.environ.get("WSPREV_LOGS", str(data.logs_dir())))
    ap.add_argument("--out", default=str(REPO / "results" / "cost_estimate.json"))
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    means = {}
    for model, folder in MODELS.items():
        for bench in BENCHES:
            texts = []
            pattern = f"{a.logs}/exp8_math/eval/{folder}/parts/{bench}_s*/{bench}_first_samples.jsonl.gz"
            for f in sorted(glob.glob(pattern)):
                with gzip.open(f, "rt", encoding="utf-8") as g:
                    texts += [json.loads(line)["text"] for line in g]
            if not texts:
                sys.exit(f"no texts at {pattern}")
            means[f"{model}/{bench}"] = per_sample(texts, tok)
            print(model, bench, json.dumps(means[f"{model}/{bench}"]), flush=True)
    out = {"method": __doc__.split("\n\n")[2].replace("\n", " "), "answer_tokens": ANSWER,
           "assumed_no_think_tokens": ASSUMED_NO_THINK, "per_sample": means, "stage_tokens": stage_totals(means)}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for k, v in out["stage_tokens"].items():
        print(f"{k:40s} {v / 1e6:8.1f}M")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
