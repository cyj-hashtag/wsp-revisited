# Pre-registration: RQ1, is the baseline of Experiment 8 fair?

**Status: REGISTERED on 2026-11-03 by cyj-hashtag (the author)** in commit
`d06c4750112c2526ebb006f78ef94c48d6e525dc`, pushed to GitHub before any benchmark generation of RQ1, including the D0
calibration run. From then on, this file changes only through dated amendments, each with its reason, listed at the
end. Amendment 1 (2026-11-03, before any generation) typeset the formulas in LaTeX and made the text more precise; it
changed no model, configuration, procedure, threshold, test or conclusion.

## Notation

| Symbol | Meaning |
|---|---|
| $M \in \lbrace \mathrm{base}, \mathrm{p50c} \rbrace$ | model (section 2) |
| $c \in \lbrace \mathrm{D0}, \mathrm{D1}, \mathrm{D2a}, \mathrm{D2b}, \mathrm{D3} \rbrace$ | decoding configuration (section 3) |
| $i = 1, \dots, N$ | problem of a benchmark; $N = 500$ for MATH-500 |
| $s = 0, \dots, S_c - 1$ | sample of a problem, generated with seed $s$; $S_c$ samples per problem (section 4) |
| $y_{M,c,i,s} \in \lbrace 0, 1 \rbrace$ | 1 if the sample is graded correct |
| $a_{M,c,i} = \frac{1}{S_c} \sum_{s} y_{M,c,i,s}$ | accuracy on problem $i$ |
| $\operatorname{Acc}(M, c) = \frac{1}{N} \sum_{i} a_{M,c,i}$ | accuracy, the mean over all samples |
| $\Delta_c = \operatorname{Acc}(\mathrm{p50c}, c) - \operatorname{Acc}(\mathrm{base}, c)$ | edit effect under $c$ |

Accuracies and effects are fractions; one percentage point (pp) is $0.01$.

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

Defined in `src/wsprev/decoding.py` at the registration commit. Common to all configurations: `min_p` 0, repetition
penalty 1.0, at most $L = 32768$ new tokens per request, vLLM 0.33.0 in bfloat16 with `max_model_len` of
$L + 2048 = 34816$, and the model's chat template with `enable_thinking` as below. The user message is the problem,
then `"\n"`, then `"Please reason step by step, and put your final answer within \boxed{}."` (IFEval: its prompt).
Sample $s$ of a problem is its own request with seed $s$.

| Name | Thinking | Temperature $T$ | `top_p` | `top_k` | Presence penalty | Source |
|---|---|---|---|---|---|---|
| D0 | on | 0.6 | 0.95 | 20 | 1.5 | the original think35 |
| D1 | on | 1.0 | 0.95 | 20 | 1.5 | model card: thinking, text tasks; its language benchmarks |
| D2a | off | 1.0 | 1.00 | 20 | 2.0 | model card: non-thinking, text tasks |
| D2b | off | 0.7 | 0.80 | 20 | 1.5 | model card: non-thinking benchmark setting |
| D3 | on | 0.6 | 0.95 | 20 | 1.5 | D0 + loop breaker (below) |

### D3: loop breaker

Let $x_1, x_2, \dots$ be the thinking tokens of one uninterrupted request with D0's sampling, the same request as under
D0. Checkpoint $t \in \lbrace 1024, 2048, 3072, \dots \rbrace$ is reached when the thinking has $t$ tokens and the
model has not yet written `</think>`. At checkpoint $t$ the breaker computes the duplicate share of 8-grams among the
last $W = \min(t, 2048)$ thinking tokens, one minus the number of distinct 8-grams over the number of 8-gram
positions:

$$\mathrm{dup}_8(t) = 1 - \frac{\left\lvert \left\lbrace (x_j, \dots, x_{j+7}) : t - W + 1 \le j \le t - 7 \right\rbrace \right\rvert}{W - 7}.$$

1. The breaker fires at the first checkpoint with $\mathrm{dup}_8(t) \ge \tau$ or $t \ge 30720$.
2. It then keeps $x_1, \dots, x_t$, appends the closing string `\n</think>\n\n` (3 tokens), and samples at most 2,048
   answer tokens with the same sampling.
3. The presence penalty keeps counting the kept thinking: every token id that occurs in $x_1, \dots, x_t$ or in the
   closing string gets logit bias $-1.5$, and the answer request's own presence penalty is 0. This differs from one
   uninterrupted generation only for tokens that occur in the answer but nowhere in the thinking: they are not
   penalized when repeated.
4. The breaker acts only on a running request: a request that ends by itself (end of text, or the token limit) in the
   step that reaches a checkpoint is recorded as it is.
5. Every record stores whether and where the breaker fired, the reason ($\mathrm{dup}_8$ or length) and
   $\mathrm{dup}_8(t)$ at that point.

A sample cut by the breaker thus has at most $30720 + 3 + 2048 = 32771$ generated tokens. It counts as cut at the
token limit if its answer stops at 2,048 tokens; a D3 sample that the breaker did not cut counts as cut if it stops at
$L$, as under D0.

### D3 threshold

Chosen before any benchmark generation with D3. The base model answers the first 256 problems of the practice DEV
split (`results/splits.json`) once under D3 with each $\tau \in \mathcal{T} = \lbrace 0.3, 0.5, 0.7 \rbrace$. With
$C(\tau)$ its number of correct answers, the chosen value is the most accurate one, ties broken toward the larger
$\tau$ (the breaker fires less):

$$\tau^{\star} = \max \left\lbrace \tau \in \mathcal{T} : C(\tau) = \max_{\tau' \in \mathcal{T}} C(\tau') \right\rbrace.$$

The value, the calibration results and the commit that sets `decoding.D3_TAU` are reported. For reference only:
Astra's own tail statistic of the base model's cut practice answers (the same formula on the last 2,500 words) had
mean 0.958.

## 4. Data and samples

| Benchmark | Problems $N$ | $S_c$ for D1, D2a, D2b, D3 | D0 |
|---|---|---|---|
| MATH-500 (primary) | 500 | 2 | $S_{\mathrm{D0}} = 1$ (the calibration run) |
| GSM8K test | 1,319 | 1 | the original evaluation's results |
| IFEval | 541 | 2 | the original evaluation's results |

Both models are run with every configuration. Grading is the original evaluation's: the last `\boxed{}` of the text
after the last `</think>` (the whole text if there is none), checked with math-verify 0.9.0 (`third_party/wsp_logs/
mathqa.py`); IFEval strict prompt-level accuracy with the official checkers on the text after the last `</think>`
(empty, so not followed, when thinking is on and never closed). Every sample counts; nothing is excluded. Grading
errors are regraded from the stored text.

**Calibration gate** (before any other benchmark generation): D0, MATH-500, one sample per problem. The run must give

$$\operatorname{Acc}(\mathrm{base}, \mathrm{D0}) \in [0.045, 0.115] \quad \text{and} \quad \operatorname{Acc}(\mathrm{p50c}, \mathrm{D0}) \in [0.43, 0.55].$$

Outside, we stop and fix the evaluation pipeline (vLLM version, chat template, enable_thinking, prompt, math-verify
version, checkpoint) and rerun D0; such fixes are reported. The gate does not change any criterion below.

## 5. Hypotheses and tests

Primary metric: MATH-500 accuracy $\operatorname{Acc}(M, c)$ over all samples. With the margin $m = 0.05$ (5 pp):

| | Hypothesis | Reading |
|---|---|---|
| H1 (replication) | $\Delta_{\mathrm{D0}} \ge 0.30$ | point estimate; reported with its 95% CI |
| H2 | $\Delta_{\mathrm{D1}} \lt \Delta_{\mathrm{D0}}$ | the gain is smaller with the model card's thinking setting |
| H3 | $\Delta_{\mathrm{D3}} \lt m$ | with the loop breaker the edit adds less than $m$ |
| H4 | $\Delta_{\mathrm{D2b}} \lt m$ | in the model's default non-thinking mode the edit adds less than $m$ |

**Bootstrap.** Problems are the resampling unit, paired over models and configurations. For $b = 1, \dots, B$ with
$B = 10000$, draw problem indices $i^{(b)}_1, \dots, i^{(b)}_N$ independently and uniformly from
$\lbrace 1, \dots, N \rbrace$ (one draw from `numpy.random.default_rng(0)`, shared by every model and configuration),
and set

$$\operatorname{Acc}^{(b)}(M, c) = \frac{1}{N} \sum_{j=1}^{N} a_{M, c, i^{(b)}_j}, \qquad \Delta^{(b)}_c = \operatorname{Acc}^{(b)}(\mathrm{p50c}, c) - \operatorname{Acc}^{(b)}(\mathrm{base}, c).$$

A 95% confidence interval runs from the 2.5th to the 97.5th percentile of the $B$ replicates.

**Tests.** One-sided bootstrap p-values:

$$p_2 = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\left[ \Delta^{(b)}_{\mathrm{D0}} - \Delta^{(b)}_{\mathrm{D1}} \le 0 \right]$$

$$p_3 = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\left[ \Delta^{(b)}_{\mathrm{D3}} \ge m \right]$$

$$p_4 = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\left[ \Delta^{(b)}_{\mathrm{D2b}} \ge m \right]$$

Holm's correction over H2-H4 at family-wise $\alpha = 0.05$: with $p_{(1)} \le p_{(2)} \le p_{(3)}$ the sorted p-values,
the adjusted values are

$$\tilde{p}_{(j)} = \max_{l \le j} \min\left( 1, (4 - l) p_{(l)} \right),$$

and a hypothesis is supported if its adjusted p-value is below $\alpha$. For H3 and H4 this requires the data to rule
out an effect of $m$ or more, so an underpowered run cannot support them.

**Cross comparison (secondary).**

$$\Delta_{\times} = \operatorname{Acc}(\mathrm{p50c}, \mathrm{D0}) - \operatorname{Acc}(\mathrm{base}, c^{\star}),$$

where $c^{\star}$ is the base configuration with the highest accuracy among D1, D2b and D3 on the practice SELECT split
(128 problems, one sample each; ties: D1, then D2b, then D3), reported with its 95% CI from the same replicates.
$c^{\star}$ is fixed before the benchmark runs; the benchmark never takes part in the choice.

**Reported for every model, configuration and benchmark:** accuracy, share of answers cut at the token limit,
accuracy of finished and of cut answers, mean generated tokens, share of thinking closed by the model, breaker share
and position (D3), and the paired accuracy on problems where both models finish, that is, where no sample of either
model was cut. Every configuration that was run goes into the appendix, whatever its result. GSM8K and IFEval are
secondary: the effect $\Delta_c$ of every configuration with its 95% CI from the same bootstrap over that benchmark's
problems (for D0, from the original evaluation's per-problem results), without hypothesis tests.

## 6. Conclusions (fixed wording)

- H3 and H4 both supported: "The edit's gain on MATH-500 mainly reflects the unedited model's failure to stop
  thinking; decoding-level remedies give the unedited model most of it."
- Neither supported: "The edit retains a substantial gain under the fair baselines we tested."
- Otherwise: the effect sizes are reported as they are, without either summary sentence.

## 7. Software

Generation, grading and analysis code: this repository at the registration commit (`src/wsprev/`, `scripts/`).
Any later change to code that affects RQ1 results is listed in an amendment.

## Amendments

### Amendment 1 (2026-11-03)

Made at the author's request after reviewing the registered text, before any generation on benchmark or practice
problems; no GPU had been rented yet. The text as registered is in commit `d06c4750112c2526ebb006f78ef94c48d6e525dc`;
`git log -p -- prereg/criteria.md` shows every change since.

1. **Presentation.** Formulas are typeset in LaTeX; a notation table was added; the D3 rule, the threshold choice, the
   calibration gate, the hypotheses and the statistics are written as formulas. Section numbers are unchanged.
2. **Clarifications.** They make the registered procedure explicit; none changes it.
   - The bootstrap draws one matrix of problem indices from `numpy.random.default_rng(0)` and uses it for every model
     and configuration; confidence intervals are percentile intervals, the cross comparison's included. This is what
     `src/wsprev/rq1.py` does at the registration commit.
   - The closing string `\n</think>\n\n` is 3 tokens of the model's tokenizer, so a sample cut by the D3 breaker can
     have up to 32,771 generated tokens, not 32,768; it counts as cut at the token limit if its answer stops at 2,048
     tokens. This is what `src/wsprev/generate.py` does at the registration commit.
   - The registered sentence "An answer that ends by itself (end of text) is recorded as it is" means that the breaker
     acts only on a running request (D3, item 4), as `src/wsprev/generate.py` does at the registration commit.
   - "Problems where both models finish" are those where no sample of either model was cut.
   - The secondary effects on GSM8K and IFEval cover every configuration; for D0 they use the original evaluation's
     per-problem results, which section 4 names as the D0 data.
3. **Analysis code.** At the registration commit, `src/wsprev/rq1.py` did not compute the secondary effects that
   section 5 registers. Commit `5b5edf2f07eb8e916457b228f24e9be1597c07b7` adds them, with the bootstrap above and no
   tests; the registered tests are unchanged. No other code that affects RQ1 results has changed.
