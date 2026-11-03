# Pre-registration: RQ1, is the baseline of Experiment 8 fair?

**Status: DRAFT, not registered.** Registration is the commit that replaces this line with
"REGISTERED on <date> by <author>"; that commit is pushed before any benchmark generation of RQ1, including the D0
calibration run. After it, this file is only changed by dated amendments appended at the end, each with its reason.

> **Decisions for the author before registration** (delete this box in the registration commit):
> 1. Margins: H1 uses 30 percentage points (pp), H3 and H4 use m = 5 pp. Keep or change.
> 2. H3 / H4 criterion. Proposed here: "the upper end of the Holm-adjusted one-sided confidence bound of the edit effect
>    is below m" (a non-inferiority test, which an underpowered run cannot pass by accident). The plan's first draft
>    was weaker: "the point estimate is below m, or the 95% CI contains 0". Choose one.
> 3. D3 implementation (section 3): one uninterrupted generation watched by the breaker, and the presence penalty
>    carried into the answer. This replaces the plan's "generate 1,024 tokens at a time and continue with prefix
>    caching", which would reset vLLM's presence penalty every 1,024 tokens and so change D0's sampling. Confirm.
> 4. tau calibration on the first 256 problems of DEV (base model, one sample each). Confirm the size.
> 5. Base configuration for the cross comparison chosen among {D1, D2b, D3} (as in the plan's formula), not all five.

## 1. Question

Experiment 8 of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, 2026) reports that an edit of
Qwen3.5-0.8B written by GPT-6 Astra (run p50c) raises MATH-500 accuracy from 7.9% to 48.1% under the evaluation
setting "think35" (thinking on, temperature 0.6, presence penalty 1.5, at most 32,768 new tokens), in which 93.0% of
the unedited model's answers reach the token limit. We ask how much of the edit's gain remains when the unedited model
is decoded with the settings of its model card, or with a simple decoding-level fix for thinking that does not end.

## 2. Models

- **base**: `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`; weight file sha256
  `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (checked by `scripts/verify_model.py`).
- **p50c**: base with the edit `exp8_math/runs/p50c/edit/` of `RikiyaT/lm_understand-logs` at commit
  `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, rebuilt by `scripts/build_p50c.py` with the authors' `apply_edit.rebuild`;
  both edited tensors match the sha256 in the edit's `manifest.json`, and every other byte equals the base file.

## 3. Decoding configurations

Defined in `src/wsprev/decoding.py` at the registration commit. All: min_p 0, repetition penalty 1.0, at most 32,768
new tokens, vLLM 0.33.0 in bfloat16 with max_model_len 34,816, the model's chat template with `enable_thinking` as
below, user message = problem + "\n" + "Please reason step by step, and put your final answer within \boxed{}."
(IFEval: its prompt). Sample k of a problem is its own request with seed k.

| Name | Thinking | Temperature | top_p | top_k | Presence penalty | Source |
|---|---|---|---|---|---|---|
| D0 | on | 0.6 | 0.95 | 20 | 1.5 | the original think35 |
| D1 | on | 1.0 | 0.95 | 20 | 1.5 | model card: thinking, text tasks; its language benchmarks |
| D2a | off | 1.0 | 1.00 | 20 | 2.0 | model card: non-thinking, text tasks |
| D2b | off | 0.7 | 0.80 | 20 | 1.5 | model card: non-thinking benchmark setting |
| D3 | on | 0.6 | 0.95 | 20 | 1.5 | D0 + loop breaker (below) |

**D3, loop breaker.** The thinking is generated as one uninterrupted request with D0's sampling, the same request
as under D0. When the thinking reaches t tokens, for t = 1,024, 2,048, ..., before the model writes `</think>`, the
breaker computes dup8 = 1 − (distinct 8-grams) / (8-gram positions) over the last 2,048 thinking token ids. It fires if
dup8 ≥ tau or t ≥ 30,720. It then keeps the first t thinking tokens, appends `\n</think>\n\n` and samples at most 2,048
answer tokens with the same sampling; the presence penalty keeps counting the kept thinking (every token id in it gets
logit bias −1.5 and the answer's own presence penalty is 0; this differs from one uninterrupted generation only for
tokens that occur in the answer but nowhere in the thinking). An answer that ends by itself (end of text) is recorded
as it is. Every record stores whether and where the breaker fired, the reason (dup8 or length) and dup8 at that point.

**tau.** Chosen before any benchmark generation with D3: base model, the first 256 problems of the practice DEV split
(`results/splits.json`), one sample each, D3 with tau in {0.3, 0.5, 0.7}; tau = the value with the highest accuracy,
ties broken toward the larger tau (the breaker fires less). The value, the calibration results and the commit that
sets `decoding.D3_TAU` are reported. For reference only: Astra's own tail statistic of the base model's cut practice
answers (the same formula on the last 2,500 words) had mean 0.958.

## 4. Data and samples

| Benchmark | Problems | Samples per problem: D1, D2a, D2b, D3 | D0 |
|---|---|---|---|
| MATH-500 (primary) | 500 | 2 | 1 (the calibration run) |
| GSM8K test | 1,319 | 1 | the original evaluation's results |
| IFEval | 541 | 2 | the original evaluation's results |

Both models are run with every configuration. Grading is the original evaluation's: the last `\boxed{}` of the text
after the last `</think>` (the whole text if there is none), checked with math-verify 0.9.0 (`third_party/wsp_logs/
mathqa.py`); IFEval strict prompt-level accuracy with the official checkers on the text after the last `</think>`
(empty, so not followed, when thinking is on and never closed). Every sample counts; nothing is excluded. Grading
errors are regraded from the stored text.

**Calibration gate** (before any other benchmark generation): D0, MATH-500, one sample per problem; base must fall in
[4.5%, 11.5%] and p50c in [43%, 55%]. Outside, we stop and fix the evaluation pipeline (vLLM version, chat template,
enable_thinking, prompt, math-verify version, checkpoint) and rerun D0; such fixes are reported. The gate does not
change any criterion below.

## 5. Hypotheses and tests

Primary metric: MATH-500 accuracy over all samples. Edit effect under configuration c:
Δ_c = Acc(p50c, c) − Acc(base, c), for c in {D0, D1, D2b, D3}.

- **H1 (replication).** Δ_D0 ≥ 30 pp (point estimate; reported with its 95% CI).
- **H2.** Δ_D1 < Δ_D0: the gain is smaller with the model card's thinking setting.
- **H3.** Δ_D3 < m: with the loop breaker the edit adds less than m.
- **H4.** Δ_D2b < m: in the model's default non-thinking mode the edit adds less than m.

m = 5 pp. Statistics: problems are the resampling unit (paired over models and configurations), 10,000 bootstrap
resamples with numpy seed 0. One-sided bootstrap p-values: H2, p = P*(Δ*_D0 − Δ*_D1 ≤ 0); H3, p = P*(Δ*_D3 ≥ m);
H4, p = P*(Δ*_D2b ≥ m). Holm's correction over H2-H4 at family-wise α = 0.05; a hypothesis is supported if its
adjusted p-value is below 0.05.

**Cross comparison (secondary).** Acc(p50c, D0) − Acc(base, c*), where c* is the base configuration with the highest
accuracy among {D1, D2b, D3} on the practice SELECT split (128 problems, one sample each; ties: D1, then D2b, then D3).
c* is fixed before the benchmark runs; the benchmark never takes part in the choice.

**Reported for every model × configuration × benchmark:** accuracy, share of answers cut at the token limit,
accuracy of finished and of cut answers, mean generated tokens, share of thinking closed by the model, breaker share
and position (D3), and the paired accuracy on problems where both models finish. Every configuration that was run goes
into the appendix, whatever its result. GSM8K and IFEval are secondary: effects with 95% CIs, no hypothesis tests.

## 6. Conclusions (fixed wording)

- H3 and H4 both supported: "The edit's gain on MATH-500 mainly reflects the unedited model's failure to stop
  thinking; decoding-level remedies give the unedited model most of it."
- Neither supported: "The edit retains a substantial gain under the fair baselines we tested."
- Otherwise: the effect sizes are reported as they are, without either summary sentence.

## 7. Software

Generation, grading and analysis code: this repository at the registration commit (`src/wsprev/`, `scripts/`).
Any later change to code that affects RQ1 results is listed in an amendment.

## Amendments

(none)
