"""GPU preflight: checks, with the base model and in a few minutes, the vLLM behaviour wsprev.generate relies on.
Run it once on a new machine, after scripts/setup.sh and before the smoke test.

    python scripts/check_vllm.py [--model base] [--out results/preflight.json]

  1  GPU visible to torch; library versions
  2  tokenizer: a single </think> token contained in CLOSE_THINK; the D0 prompt ends with "<think>\\n"
  3  grading in the main thread (math-verify's timeouts use signal.alarm)
  4  logit_bias is applied (the greedy first token, banned, changes)
  5  aborting a running request ends it cleanly
  6  seeds: one request per sample with seed s + k gives the same tokens as one request with n samples and seed s
     (if not, D0's samples do not reuse the original evaluation's random streams: a warning, not an error)
  7  the D3 breaker end to end (tau = 0, so it fires at the first checkpoint unless the thinking closes before)
  8  logit_bias acts on the logits before temperature, as the presence penalty does; D3 carries the presence penalty
     into the answer as a logit bias (prereg section 3), which is equivalent only then
Any FAIL: do not start the smoke test; send the output to Claude. Exit status 1 on FAIL.
"""
import argparse
import json
import math
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wsprev import data, decoding, grade  # noqa: E402
from wsprev.generate import Runner, VLLMEngine, gpu_info, record_of, resolve_model, versions  # noqa: E402

RESULTS = []


def report(check, status, detail=""):
    RESULTS.append({"check": check, "status": status, "detail": detail})
    print(f"[{status}] {check}: {detail}", flush=True)


def collect(engine, rids, timeout=600):
    """Step the engine until the requests finish; {rid: generated token ids}."""
    toks, done, t0 = {r: [] for r in rids}, set(), time.time()
    while not set(rids) <= done:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"requests {sorted(set(rids) - done)} did not finish in {timeout} s")
        for d in engine.step():
            if d.rid in toks:
                toks[d.rid] += d.token_ids
                if d.finished:
                    done.add(d.rid)
    return toks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--out", default=str(REPO / "results" / "preflight.json"))
    a = ap.parse_args()

    import torch

    gpu = gpu_info()
    report("1 GPU", "PASS" if torch.cuda.is_available() else "FAIL", f"{gpu} | {versions()}")
    if not torch.cuda.is_available():
        return finish(a.out)

    from transformers import AutoTokenizer

    name, model_dir, _ = resolve_model(a.model)
    tok = AutoTokenizer.from_pretrained(model_dir)
    think_end = tok.convert_tokens_to_ids("</think>")
    close_ids = tok.encode(decoding.CLOSE_THINK, add_special_tokens=False)
    row = data.load("math500")[0]
    prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(row)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    ok = isinstance(think_end, int) and think_end in close_ids and prompt.endswith("<|im_start|>assistant\n<think>\n")
    report("2 tokenizer and prompt", "PASS" if ok else "FAIL",
           f"</think> = {think_end}, CLOSE_THINK -> {close_ids}, prompt ends {prompt[-40:]!r}")

    ok = (grade.grade({"answer": "\\frac{1}{2}"}, "so \\boxed{\\dfrac{1}{2}}", True)["correct"]
          and not grade.grade({"answer": "2"}, "so \\boxed{3}", True)["correct"])
    report("3 grading in the main thread", "PASS" if ok else "FAIL", "1/2 = \\dfrac{1}{2} and 2 != 3")

    engine = VLLMEngine(model_dir, a.gpu_mem)
    ids = list(tok.encode(prompt))

    greedy = decoding.D0.sampling(temperature=0.0, max_tokens=4, seed=0)
    engine.add("g0", ids, greedy)
    first = collect(engine, ["g0"])["g0"][0]
    engine.add("g1", ids, dict(greedy, logit_bias={first: -100.0}))
    other = collect(engine, ["g1"])["g1"][0]
    report("4 logit_bias", "PASS" if other != first else "FAIL", f"greedy first token {first}, banned -> {other}")

    engine.add("a0", ids, decoding.D0.sampling(max_tokens=2000, seed=0))
    while not any(d.rid == "a0" and d.token_ids for d in engine.step()):
        pass
    engine.abort("a0")
    steps = 0
    while engine.has_unfinished() and steps < 2000:
        engine.step()
        steps += 1
    report("5 abort", "PASS" if not engine.has_unfinished() else "FAIL", f"engine idle after {steps} steps")

    from vllm import SamplingParams

    sp = decoding.D0.sampling(max_tokens=32)
    ref = engine.llm.generate([{"prompt_token_ids": ids}], SamplingParams(n=2, seed=0, **sp), use_tqdm=False)
    ref = [list(o.token_ids) for o in sorted(ref[0].outputs, key=lambda o: o.index)]
    engine.add("s0", ids, dict(sp, seed=0))
    engine.add("s1", ids, dict(sp, seed=1))
    mine = collect(engine, ["s0", "s1"])
    same = ref == [mine["s0"], mine["s1"]]
    report("6 seeds", "PASS" if same else "WARN",
           "n=2/seed 0 equals seeds 0 and 1" if same else
           "differs: D0 samples will not reuse the original evaluation's random streams (allowed; note it)")

    cfg = replace(decoding.d3(0.0), name="D3@0")
    recs = []
    runner = Runner(engine, tok, cfg, lambda s, r: recs.append(record_of(s, cfg, len(close_ids), r, {})),
                    log=lambda *x, **k: None)
    rows = data.load("math500")[:2]
    t0 = time.time()
    runner.run([(rows[0], 0), (rows[1], 0)])
    ok, details = len(recs) == 2, []
    for rec, text in recs:
        if rec["loop_triggered"]:
            ok &= (rec["trigger_pos"] == cfg.breaker.every and "</think>" in text
                   and rec["n_gen_tokens"] == rec["trigger_pos"] + len(close_ids) + rec["answer_tokens"]
                   and rec["answer_tokens"] <= cfg.breaker.answer_tokens)
        else:
            ok &= bool(rec["think_closed"]) and rec["think_tokens"] < cfg.breaker.every
        details.append({k: rec[k] for k in ("problem_id", "loop_triggered", "trigger_pos", "answer_tokens",
                                             "n_gen_tokens", "finish_reason", "think_closed")})
    report("7 D3 breaker end to end", "PASS" if ok else "FAIL", f"{details} in {time.time() - t0:.0f} s")

    # 8: the raw logit lead d of the most likely first token x over the second y, then 2,000 one-token samples at
    # temperature T with logit bias b = -(d + 1) on x; the two orders predict log(p_x / p_y) values 1 + d apart
    T = 0.5
    raw = engine.llm.generate([{"prompt_token_ids": ids}],
                              SamplingParams(temperature=1.0, max_tokens=1, logprobs=5, seed=0), use_tqdm=False)
    (x, lx), (y, ly) = sorted(((t, lp.logprob) for t, lp in raw[0].outputs[0].logprobs[0].items()),
                              key=lambda kv: -kv[1])[:2]
    d = lx - ly
    b = max(-100.0, -(d + 1.0))
    out = engine.llm.generate([{"prompt_token_ids": ids}],
                              SamplingParams(n=2000, temperature=T, top_p=1.0, max_tokens=1, seed=0,
                                             logit_bias={int(x): b}), use_tqdm=False)
    toks = [o.token_ids[0] for o in out[0].outputs]
    report("8 logit_bias before temperature", *bias_order_verdict(d, b, T, toks.count(x), toks.count(y)))
    return finish(a.out, {"model": name, "gpu": gpu, "versions": versions(), "host": platform.node()})


def bias_order_verdict(d, b, T, nx, ny):
    """(status, detail) of check 8. With raw logit lead d of token x over token y, logit bias b on x and temperature
    T, log(p_x / p_y) is (d + b) / T if the bias acts before temperature and d / T + b if it acts after."""
    before, after = (d + b) / T, d / T + b
    r = math.log((nx + 0.5) / (ny + 0.5))
    se = math.sqrt(1 / (nx + 0.5) + 1 / (ny + 0.5))
    detail = (f"log(n_x/n_y) = {r:.2f} +- {se:.2f} (n_x {nx}, n_y {ny}); predicted {before:.2f} if the bias acts "
              f"before temperature, {after:.2f} if after")
    if abs(r - before) < abs(r - after) and abs(r - before) <= 3 * se + 0.3:
        return "PASS", detail
    if abs(r - after) < abs(r - before):
        return "FAIL", detail + " -> D3's carried presence penalty must be scaled by 1/T; fix before stage tau"
    return "WARN", detail


def finish(out, info=None):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"info": info, "checks": RESULTS}, indent=1) + "\n", encoding="utf-8")
    failed = [r["check"] for r in RESULTS if r["status"] == "FAIL"]
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'All required checks passed.'} -> {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
