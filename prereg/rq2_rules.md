# RQ2 rules: how the LLM-free edit is chosen

**Status: written 2026-11-05, before any RQ2 generation.** RQ2 has no hypotheses and is not part of the
registered RQ1 analysis (`prereg/criteria.md`). This file fixes the procedure that chooses the edit, before any data
that it uses exists, so that the choice cannot be tuned to the results. Its commit is reported with the results; from
then on it changes only through dated amendments at the end, each with its reason. The code is `src/wsprev/rq2.py`
and `src/wsprev/directions.py` at that commit; the procedure below and the code say the same thing, and where they
differ the difference is reported as an amendment.

## 1. Question

Can a fixed procedure that never calls an LLM, never uses a benchmark problem, and edits the same kind of weights as
Experiment 8's p50c (two MLP `down_proj` matrices, a column-masked rank-one change), find an edit whose effect on
MATH-500 compares with p50c's?

## 2. Data

All choices use the practice problems of Experiment 8 (GSM8K train and MATH train) split by `results/splits.json`
(seed 20261103): DEV (512), HOLDOUT (256), SELECT (128), disjoint. Benchmarks are used once, after the choice (section
7). Within DEV, problems are taken in the split's order:

| Use | Problems |
|---|---|
| Directions (section 3) | DEV[0:`dev_problems`] |
| Screening (section 5) | DEV[448:512], never used for directions |
| Final choice (section 6) | HOLDOUT[0:`holdout_problems`] |
| Swap check (section 5) | SELECT[0:4], greedy, 48 tokens, no grading |

## 3. Directions and neuron scores

1. **Generation.** The base model answers DEV[0:`dev_problems`] once under D0 (`src/wsprev/decoding.py`; seed 0,
   at most 32,768 tokens), as in RQ1.
2. **Windows** (2,048 tokens; dup8 as in RQ1's D3: one minus the distinct 8-grams over the 8-gram positions). For
   every response that was cut at the token limit with its thinking never closed, and has at least 4,096 thinking
   tokens: the **loop** window is its last 2,048 thinking tokens and the **normal** window its first 2,048. The pair is
   used if dup8(loop) >= 0.8 and dup8(normal) <= 0.3; the first `max_pairs` such pairs in DEV order are used. Control
   windows: the middle 2,048 thinking tokens of every response that ended its thinking and finished, if it has at
   least 2,048 thinking tokens. Thinking tokens are the response text before the first `</think>`, re-tokenized.
3. **Reading.** Teacher forcing with HF transformers 5.20.2 (bfloat16, SDPA attention, no gradients) on the prompt
   followed by the thinking up to the window's end. For each layer l in 8, ..., 22 (0-indexed decoder layers) the
   means over the window's 2,048 positions of the residual stream at layer l's output and of the input of layer l's
   `mlp.down_proj` are kept. Only if the full context does not fit in GPU memory, contexts are cut from the left to
   the length that fits, keeping the window; this is reported.
4. **Direction.** u_l is the mean over pairs of (loop mean - normal mean) at layer l's output, normalized to unit
   length.
5. **Layers.** Cross-fitted AUC: u from the even-numbered pairs scores the odd ones (AUC of loop vs normal
   projections) and the reverse; the two AUCs are averaged. The **AUC layers** are the two with the highest AUC, ties
   broken by the larger standardized separation of the projections, then by the lower layer index. **Astra's layers**
   {15, 19} form the other layer set; if the AUC layers are {15, 19} there is one set.
6. **Scores.** c_j = (u_l^T w_j)(E_loop[a_j] - E_normal[a_j]) for the columns w_j of layer l's `down_proj` weight W_l
   (base model) and the mean `down_proj` input a over the loop and the normal windows. The top-k columns are the k
   largest c_j (ties: lower index).

## 4. The edits

For a layer set, k and alpha, every layer l of the set gets
W_l' = W_l - alpha u_l (m_k ⊙ (u_l^T W_l)), with m_k the 0/1 mask of its top-k columns (k = 3,584 is every column),
computed in float64 from the bfloat16 base weights, converted to float32 and rounded to bfloat16 (nearest even), as
the authors' `st_numpy.bf16_bits` does. The grid is every
combination of the layer sets, k and alpha of the plan (section 8).

## 5. Screening

Before screening, the **swap check** must pass: the screening engine replaces `down_proj` weights in place
(`src/wsprev/_vllm_swap.py`); with p50c's two tensors swapped into the base engine, greedy outputs on the check prompts
must equal those of p50c loaded from disk, and after restoring, those of the base model, with the weights
bit-identical to the base file.

Every edit of the grid answers the 64 screening problems once under D0 with at most 8,192 tokens (seed 0). Grading is
RQ1's; every sample counts. The base model and p50c are run the same way for reference and are never chosen. Edits
are ranked by: more correct answers; then more answers not cut at the limit; then smaller alpha; then smaller k; then
Astra's layer set before the AUC set. The first `top` edits go to the final choice.

## 6. Final choice

Each of them answers HOLDOUT[0:`holdout_problems`] once under D0 (at most 32,768 tokens); the same ranking picks one
edit, **rq2-auto**. It is written as a model folder whose weight file equals the base model's except for the edited
tensors (checked byte by byte).

## 7. Evaluation (once, after the choice; no hypothesis tests)

On MATH-500, as in RQ1: rq2-auto under D0 (1 sample per problem) and, in the full plan, D3 (2 samples). It is
compared with RQ1's base and p50c under the same configuration: differences in accuracy with 95% percentile intervals
from a paired bootstrap over problems (10,000 resamples, `numpy.random.default_rng(0)`). Also reported: cut share,
mean tokens, every screening and HOLDOUT result, the AUC of every layer, the first-order check of Proposition 3 for
every edit, and the cost (GPU hours and generated tokens; LLM tokens: 0).

**Controls**, MATH-500 under D0, 1 sample, the first `control_problems` problems, each compared with base on the same
problems (seed 20261105 for the random draws):

- a: rq2-auto's columns and alpha with a uniformly random unit direction per layer;
- b: rq2-auto's direction and alpha with k random columns per layer (not run if rq2-auto changes every column);
- c: p50c's columns and direction with alpha = 1 instead of 6 (a projection, Proposition 1), on layers 15 and 19.
  p50c's edit is stored only as changed bfloat16 entries, so its columns are the 64 that changed and its direction is
  the leading left singular vector of the change on them; rebuilding with alpha = 6 is reported as a check.

## 8. Plans

One plan is chosen before any RQ2 generation and committed as `results/rq2/plan.json`
(`python -m wsprev.rq2 plan lean|full`). Cost estimates: `results/rq2/cost_estimate.json`.

| | lean | full |
|---|---|---|
| `dev_problems` | 160 | 448 |
| `max_pairs` | 96 | 256 |
| k | 64, 3584 | 32, 64, 128, 3584 |
| alpha | 1, 4, 6 | 1, 2, 4, 6, 8 |
| edits (two layer sets) | 12 | 40 |
| `top` | 2 | 5 |
| `holdout_problems` | 128 | 256 |
| MATH-500 runs of rq2-auto | D0 | D0, D3 |
| `control_problems` | 100 | 500 |

## Amendments

### Amendment 1 (2026-11-06)

Two fixes of the screening engine's code after the swap check (section 5) failed on the GPU machine, before any
generation with an edit. No rule changes; the stages rq2-dev and rq2-read had finished and are unaffected.

1. vLLM 0.33.0's RPC does not rebuild a tensor inside an untyped argument: the worker received a list and the check
   stopped with an error (2026-11-06 02:28 UTC+8). The weights now reach the worker as paths of `.npy` files, and the
   restore check compares sha256 strings (commit `721e715`).
2. With the swap working, greedy outputs of p50c swapped in still differed from p50c loaded from disk on 2 of the 4
   check prompts (from token 26 and 29), while teacher-forced logprobs of the same continuations were equal at all
   192 positions. The prefix cache had kept the prompts' KV computed with the base weights:
   `LLM.reset_prefix_cache()` does not clear blocks still in use and reports this only in its return value. The
   screening engine now runs without prefix caching. Every screening and HOLDOUT prompt is distinct, so this changes
   only memory reuse, not what is computed; the MATH-500 runs (section 7) use `wsprev.generate` with RQ1's settings.

The swap check is run again with both fixes; screening starts only if it passes.

### Amendment 2 (2026-11-06)

The swap check failed again with both fixes (07:01 UTC+8), before any generation with an edit: greedy outputs still
differed from p50c loaded from disk. Prefill computes the same values with the swapped weights (teacher-forced
logprobs equal at every position), so the decode steps, which vLLM replays from CUDA graphs captured at start-up, are
the likely place where the swapped weights are not seen; this was not investigated further. The in-place swap is
dropped: in screening (section 5) and on HOLDOUT (section 6) every edit is written as a model folder, as section 6
already did for the chosen edit, and generated by `wsprev.generate`, the code of RQ1 and of section 7, one vLLM
engine per edit. The swap check is therefore no longer needed; its failed result stays in
`results/rq2/swapcheck.json`. The screening runs are D0 with at most 8,192 tokens (`D0-8k` in
`src/wsprev/decoding.py`) on DEV[448:512] (`practice:dev-screen` in `src/wsprev/data.py`), as section 5 states. No
rule changes.
