# RQ3 results: survival and decomposition

Computed by `python -m wsprev.survival` on 2026-11-05 16:31 (git 6f771db) from the records of the GPU run (`runs/gpushare_2026-11-04/runs_records.tgz`, unpacked into `runs/`). Definitions: the docstring of `src/wsprev/survival.py`. Intervals: 95% bootstrap over problems (10,000 resamples, `default_rng(0)`, problems in sorted order). Percentage points (pp) are signed.

## How the thinking ended

| Model | Config | Benchmark | Samples | `</think>` | end of text, no `</think>` | breaker | cut |
|---|---|---|---|---|---|---|---|
| base | D0 | math500 | 500 | 37 | 0 | 0 | 463 |
| p50c | D0 | math500 | 500 | 419 | 1 | 0 | 80 |
| base | D1 | math500 | 1000 | 207 | 0 | 0 | 793 |
| p50c | D1 | math500 | 1000 | 354 | 21 | 0 | 625 |
| base | D2a | math500 | 1000 | (thinking off) | 999 | | 1 |
| p50c | D2a | math500 | 1000 | (thinking off) | 1000 | | 0 |
| base | D2b | math500 | 1000 | (thinking off) | 850 | | 150 |
| p50c | D2b | math500 | 1000 | (thinking off) | 995 | | 5 |
| base | D3 | math500 | 1000 | 69 | 0 | 931 | 0 |
| p50c | D3 | math500 | 1000 | 649 | 0 | 351 | 0 |
| base | D1 | gsm8k | 1319 | 508 | 0 | 0 | 811 |
| p50c | D1 | gsm8k | 1319 | 929 | 38 | 0 | 352 |
| base | D2a | gsm8k | 1319 | (thinking off) | 1319 | | 0 |
| p50c | D2a | gsm8k | 1319 | (thinking off) | 1319 | | 0 |
| base | D2b | gsm8k | 1319 | (thinking off) | 1302 | | 17 |
| p50c | D2b | gsm8k | 1319 | (thinking off) | 1319 | | 0 |
| base | D3 | gsm8k | 1319 | 160 | 0 | 1159 | 0 |
| p50c | D3 | gsm8k | 1319 | 1154 | 0 | 165 | 0 |
| base | D1 | ifeval | 1082 | 901 | 2 | 0 | 179 |
| p50c | D1 | ifeval | 1082 | 1048 | 21 | 0 | 13 |
| base | D2a | ifeval | 1082 | (thinking off) | 1082 | | 0 |
| p50c | D2a | ifeval | 1082 | (thinking off) | 1081 | | 1 |
| base | D2b | ifeval | 1082 | (thinking off) | 1073 | | 9 |
| p50c | D2b | ifeval | 1082 | (thinking off) | 1082 | | 0 |
| base | D3 | ifeval | 1082 | 280 | 0 | 802 | 0 |
| p50c | D3 | ifeval | 1082 | 1051 | 0 | 31 | 0 |

## Kaplan–Meier: share still thinking, $\hat S(t)$

Thinking on: $T$ = thinking tokens, censored at the token limit (D0, D1) or at the breaker (D3); "/breaker-as-end" counts the breaker's cut as an end. Thinking off: $T$ = answer tokens. RMST = $\int_0^{32768} \hat S(t)\,dt$ (tokens).

| Model / config / benchmark | Samples | Events | $\hat S(2048)$ | $\hat S(4096)$ | $\hat S(8192)$ | $\hat S(16384)$ | $\hat S(30720)$ | Median | RMST |
|---|---|---|---|---|---|---|---|---|---|
| base/D0/math500 | 500 | 37 | 98.4% | 95.8% | 93.6% | 92.6% | 92.6% | not reached | 30,690 |
| p50c/D0/math500 | 500 | 420 | 96.0% | 92.8% | 83.4% | 57.0% | 19.4% | 18,351 | 18,812 |
| base/D1/math500 | 1000 | 207 | 99.1% | 97.9% | 95.6% | 89.3% | 79.9% | not reached | 29,267 |
| p50c/D1/math500 | 1000 | 375 | 98.4% | 96.1% | 91.1% | 79.0% | 64.7% | not reached | 26,383 |
| base/D2a/math500 | 1000 | 999 | 77.4% | 67.2% | 51.4% | 13.8% | 0.6% | 8,467 | 8,814 |
| p50c/D2a/math500 | 1000 | 1000 | 81.2% | 71.7% | 52.2% | 7.7% | 0.1% | 8,511 | 8,363 |
| base/D2b/math500 | 1000 | 850 | 47.5% | 40.2% | 31.6% | 20.2% | 15.1% | 1,764 | 8,467 |
| p50c/D2b/math500 | 1000 | 995 | 49.1% | 42.2% | 32.4% | 12.3% | 0.5% | 1,911 | 6,175 |
| base/D3/math500 | 1000 | 69 | 98.8% | 97.8% | 94.0% | 87.1% | 87.1% | not reached | 29,689 |
| base/D3/math500/breaker-as-end | 1000 | 1000 | 98.8% | 92.2% | 44.7% | 6.5% | 0.0% | 8,192 | 9,408 |
| p50c/D3/math500 | 1000 | 649 | 95.7% | 91.3% | 83.3% | 60.4% | 19.3% | 19,428 | 19,186 |
| p50c/D3/math500/breaker-as-end | 1000 | 1000 | 95.7% | 91.3% | 79.9% | 48.8% | 0.0% | 16,244 | 16,489 |
| base/D1/gsm8k | 1319 | 508 | 98.0% | 95.8% | 89.5% | 77.3% | 62.5% | not reached | 25,789 |
| p50c/D1/gsm8k | 1319 | 967 | 97.8% | 92.3% | 77.5% | 56.4% | 30.2% | 20,078 | 19,715 |
| base/D2a/gsm8k | 1319 | 1319 | 47.3% | 35.8% | 20.7% | 2.0% | 0.0% | 1,776 | 4,150 |
| p50c/D2a/gsm8k | 1319 | 1319 | 54.9% | 40.6% | 20.2% | 0.5% | 0.0% | 2,609 | 4,176 |
| base/D2b/gsm8k | 1319 | 1302 | 12.2% | 8.4% | 3.8% | 1.6% | 1.3% | 442 | 1,552 |
| p50c/D2b/gsm8k | 1319 | 1319 | 12.7% | 7.1% | 3.8% | 0.2% | 0.0% | 452 | 1,265 |
| base/D3/gsm8k | 1319 | 160 | 96.9% | 94.1% | 86.1% | 60.3% | 60.3% | not reached | 23,597 |
| base/D3/gsm8k/breaker-as-end | 1319 | 1319 | 96.9% | 85.1% | 19.2% | 0.8% | 0.0% | 6,144 | 6,888 |
| p50c/D3/gsm8k | 1319 | 1154 | 90.2% | 80.6% | 55.9% | 12.3% | 0.5% | 9,032 | 9,731 |
| p50c/D3/gsm8k/breaker-as-end | 1319 | 1319 | 90.2% | 80.5% | 51.8% | 8.7% | 0.0% | 8,432 | 9,083 |
| base/D1/ifeval | 1082 | 903 | 91.3% | 49.7% | 21.4% | 16.9% | 16.6% | 4,077 | 8,926 |
| p50c/D1/ifeval | 1082 | 1069 | 75.3% | 23.8% | 5.8% | 1.5% | 1.2% | 2,778 | 3,737 |
| base/D2a/ifeval | 1082 | 1082 | 1.2% | 0.6% | 0.2% | 0.1% | 0.0% | 165 | 325 |
| p50c/D2a/ifeval | 1082 | 1081 | 0.7% | 0.2% | 0.2% | 0.1% | 0.1% | 157 | 305 |
| base/D2b/ifeval | 1082 | 1073 | 1.4% | 1.0% | 0.9% | 0.8% | 0.8% | 160 | 541 |
| p50c/D2b/ifeval | 1082 | 1082 | 0.2% | 0.1% | 0.1% | 0.0% | 0.0% | 160 | 252 |
| base/D3/ifeval | 1082 | 280 | 92.0% | 77.0% | 62.9% | 61.2% | 53.1% | not reached | 20,776 |
| base/D3/ifeval/breaker-as-end | 1082 | 1082 | 90.7% | 33.4% | 4.0% | 2.4% | 0.0% | 4,096 | 4,690 |
| p50c/D3/ifeval | 1082 | 1051 | 75.0% | 23.1% | 3.9% | 1.2% | 0.7% | 2,719 | 3,501 |
| p50c/D3/ifeval/breaker-as-end | 1082 | 1082 | 75.0% | 22.7% | 3.1% | 0.6% | 0.0% | 2,717 | 3,337 |

## Shapley decomposition of the edit effect

$\mathrm{Acc} = p\,a_F + (1-p)\,a_C$ with $p$ = share finished (not cut), $a_F$, $a_C$ = accuracy of finished and of cut answers. Termination $= (p_1-p_0)\,[(a_{F0}-a_{C0})+(a_{F1}-a_{C1})]/2$; conditional $= \bar p\,(a_{F1}-a_{F0}) + (1-\bar p)\,(a_{C1}-a_{C0})$; they add up to $\Delta$. Index 0 = base, 1 = p50c.

| Config / benchmark | $p_0$ | $p_1$ | $a_{F0}$ | $a_{F1}$ | $\Delta$ (pp) | Termination (pp) | Conditional (pp) |
|---|---|---|---|---|---|---|---|
| D0/math500 | 7.4% | 70.2% | 100.0% | 72.6% | +42.8 | +53.9 [+50.1, +57.6] | -11.1 [-13.1, -9.2] |
| D1/math500 | 19.0% | 33.1% | 97.9% | 81.9% | +8.1 | +12.6 [+10.2, +15.1] | -4.5 [-5.9, -3.2] |
| D2a/math500 | 99.9% | 100.0% | 46.0% | 41.8% | -4.2 | n/a | n/a |
| D2b/math500 | 85.0% | 99.5% | 58.0% | 52.1% | +2.3 | +7.9 [+6.6, +9.2] (61 undefined) | -5.6 [-8.9, -2.2] (61 undefined) |
| D3/math500 | 40.0% | 66.3% | 95.5% | 82.4% | +16.7 | +23.3 [+20.4, +26.1] | -6.6 [-8.4, -4.7] |
| D1/gsm8k | 35.9% | 64.4% | 93.7% | 73.4% | +13.6 | +23.7 [+21.3, +26.1] | -10.0 [-11.8, -8.3] |
| D2a/gsm8k | 100.0% | 100.0% | 48.3% | 41.1% | -7.2 | n/a | n/a |
| D2b/gsm8k | 98.7% | 100.0% | 58.8% | 56.6% | -1.5 | n/a | n/a |
| D3/gsm8k | 69.2% | 89.9% | 87.5% | 75.0% | +7.0 | +16.7 [+14.6, +18.7] | -9.7 [-11.7, -7.6] |
| D1/ifeval | 81.5% | 98.8% | 71.0% | 53.1% | -6.3 | +10.3 [+8.7, +11.9] | -16.6 [-20.0, -13.0] |
| D2a/ifeval | 100.0% | 99.9% | 47.3% | 42.5% | -4.9 | n/a | n/a |
| D2b/ifeval | 99.2% | 100.0% | 52.7% | 49.8% | -2.9 | n/a | n/a |
| D3/ifeval | 84.4% | 99.0% | 68.8% | 62.5% | -2.1 | +5.4 [+2.7, +7.4] (3 undefined) | -7.5 [-11.2, -3.3] (3 undefined) |

## Paired cells (sample $s$ of base with sample $s$ of p50c)

$\Delta = \sum_{\text{cells}} P(\text{cell})\,[\mathrm{Acc}_1 - \mathrm{Acc}_0 \mid \text{cell}]$. "Both finished" compares the models on the same pairs.

| Config / benchmark | Pairs | Cell | Share | Acc base | Acc p50c | Contribution (pp) |
|---|---|---|---|---|---|---|
| D0/math500 | 500 | both finished | 7.4% | 100.0% | 100.0% | +0.0 [+0.0, +0.0] |
| D0/math500 | 500 | only p50c finished | 62.8% | 1.3% | 69.4% | +42.8 [+38.4, +47.2] |
| D0/math500 | 500 | only base finished | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D0/math500 | 500 | both cut | 29.8% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| D0/math500 | 500 | one finished | 62.8% | 1.3% | 69.4% | +42.8 [+38.4, +47.2] |
| D1/math500 | 1000 | both finished | 16.5% | 98.8% | 94.5% | -0.7 [-1.3, -0.2] |
| D1/math500 | 1000 | only p50c finished | 16.6% | 1.8% | 69.3% | +11.2 [+9.1, +13.5] |
| D1/math500 | 1000 | only base finished | 2.5% | 92.0% | 0.0% | -2.3 [-3.4, -1.3] |
| D1/math500 | 1000 | both cut | 64.4% | 0.3% | 0.2% | -0.1 [-0.4, +0.2] |
| D1/math500 | 1000 | one finished | 19.1% | 13.6% | 60.2% | +8.9 [+6.5, +11.4] |
| D2a/math500 | 1000 | both finished | 99.9% | 46.0% | 41.8% | -4.2 [-7.3, -1.1] |
| D2a/math500 | 1000 | only p50c finished | 0.1% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| D2a/math500 | 1000 | only base finished | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2a/math500 | 1000 | both cut | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2a/math500 | 1000 | one finished | 0.1% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| D2b/math500 | 1000 | both finished | 84.7% | 58.2% | 56.9% | -1.1 [-4.2, +2.1] |
| D2b/math500 | 1000 | only p50c finished | 14.8% | 1.4% | 24.3% | +3.4 [+2.3, +4.6] |
| D2b/math500 | 1000 | only base finished | 0.3% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| D2b/math500 | 1000 | both cut | 0.2% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| D2b/math500 | 1000 | one finished | 15.1% | 1.3% | 23.8% | +3.4 [+2.3, +4.6] |
| D3/math500 | 1000 | both finished | 38.1% | 95.5% | 96.3% | +0.3 [-0.4, +1.0] |
| D3/math500 | 1000 | only p50c finished | 28.2% | 0.0% | 63.5% | +17.9 [+15.2, +20.6] |
| D3/math500 | 1000 | only base finished | 1.9% | 94.7% | 10.5% | -1.6 [-2.5, -0.8] |
| D3/math500 | 1000 | both cut | 31.8% | 0.0% | 0.3% | +0.1 [+0.0, +0.3] |
| D3/math500 | 1000 | one finished | 30.1% | 6.0% | 60.1% | +16.3 [+13.5, +19.1] |
| D1/gsm8k | 1319 | both finished | 31.3% | 95.6% | 88.6% | -2.2 [-3.1, -1.3] |
| D1/gsm8k | 1319 | only p50c finished | 33.1% | 0.7% | 58.9% | +19.3 [+17.1, +21.4] |
| D1/gsm8k | 1319 | only base finished | 4.5% | 80.0% | 3.3% | -3.5 [-4.5, -2.5] |
| D1/gsm8k | 1319 | both cut | 31.1% | 0.0% | 0.2% | +0.1 [+0.0, +0.2] |
| D1/gsm8k | 1319 | one finished | 37.6% | 10.3% | 52.2% | +15.8 [+13.3, +18.2] |
| D2a/gsm8k | 1319 | both finished | 100.0% | 48.3% | 41.1% | -7.2 [-10.2, -4.2] |
| D2a/gsm8k | 1319 | only p50c finished | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2a/gsm8k | 1319 | only base finished | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2a/gsm8k | 1319 | both cut | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2a/gsm8k | 1319 | one finished | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2b/gsm8k | 1319 | both finished | 98.7% | 58.8% | 56.9% | -1.9 [-4.6, +0.8] |
| D2b/gsm8k | 1319 | only p50c finished | 1.3% | 5.9% | 35.3% | +0.4 [+0.1, +0.8] |
| D2b/gsm8k | 1319 | only base finished | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2b/gsm8k | 1319 | both cut | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2b/gsm8k | 1319 | one finished | 1.3% | 5.9% | 35.3% | +0.4 [+0.1, +0.8] |
| D3/gsm8k | 1319 | both finished | 66.1% | 87.6% | 84.7% | -1.9 [-3.3, -0.5] |
| D3/gsm8k | 1319 | only p50c finished | 23.8% | 0.0% | 47.8% | +11.4 [+9.7, +13.1] |
| D3/gsm8k | 1319 | only base finished | 3.1% | 85.4% | 2.4% | -2.6 [-3.5, -1.7] |
| D3/gsm8k | 1319 | both cut | 7.0% | 0.0% | 1.1% | +0.1 [+0.0, +0.2] |
| D3/gsm8k | 1319 | one finished | 26.9% | 9.9% | 42.5% | +8.8 [+6.8, +10.8] |
| D1/ifeval | 1082 | both finished | 81.1% | 71.2% | 54.7% | -13.4 [-16.6, -10.2] |
| D1/ifeval | 1082 | only p50c finished | 17.7% | 5.2% | 46.1% | +7.2 [+5.4, +9.1] |
| D1/ifeval | 1082 | only base finished | 0.4% | 25.0% | 0.0% | -0.1 [-0.3, +0.0] |
| D1/ifeval | 1082 | both cut | 0.8% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| D1/ifeval | 1082 | one finished | 18.0% | 5.6% | 45.1% | +7.1 [+5.3, +9.1] |
| D2a/ifeval | 1082 | both finished | 99.9% | 47.4% | 42.5% | -4.9 [-7.9, -1.9] |
| D2a/ifeval | 1082 | only p50c finished | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2a/ifeval | 1082 | only base finished | 0.1% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| D2a/ifeval | 1082 | both cut | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2a/ifeval | 1082 | one finished | 0.1% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| D2b/ifeval | 1082 | both finished | 99.2% | 52.7% | 49.7% | -3.0 [-5.5, -0.6] |
| D2b/ifeval | 1082 | only p50c finished | 0.8% | 44.4% | 66.7% | +0.2 [+0.0, +0.5] |
| D2b/ifeval | 1082 | only base finished | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2b/ifeval | 1082 | both cut | 0.0% | n/a | n/a | +0.0 [+0.0, +0.0] |
| D2b/ifeval | 1082 | one finished | 0.8% | 44.4% | 66.7% | +0.2 [+0.0, +0.5] |
| D3/ifeval | 1082 | both finished | 84.2% | 68.7% | 64.0% | -4.0 [-6.7, -1.2] |
| D3/ifeval | 1082 | only p50c finished | 14.8% | 38.8% | 53.8% | +2.2 [+0.6, +3.8] |
| D3/ifeval | 1082 | only base finished | 0.2% | 100.0% | 0.0% | -0.2 [-0.5, +0.0] |
| D3/ifeval | 1082 | both cut | 0.8% | 44.4% | 22.2% | -0.2 [-0.6, +0.2] |
| D3/ifeval | 1082 | one finished | 15.0% | 39.5% | 53.1% | +2.0 [+0.5, +3.6] |

## Problems where neither model was cut

| Config / benchmark | Problems | base | p50c | $\Delta$ (pp) |
|---|---|---|---|---|
| D0/math500 | 37 | 100.0% | 100.0% | +0.0 [+0.0, +0.0] |
| D1/math500 | 55 | 100.0% | 99.1% | -0.9 [-2.7, +0.0] |
| D2a/math500 | 499 | 46.1% | 41.9% | -4.2 [-7.4, -1.1] |
| D2b/math500 | 375 | 61.9% | 59.6% | -2.3 [-6.1, +1.7] |
| D3/math500 | 152 | 97.4% | 98.0% | +0.7 [-1.0, +2.3] |
| D1/gsm8k | 413 | 95.6% | 88.6% | -7.0 [-9.9, -4.4] |
| D2a/gsm8k | 1319 | 48.3% | 41.1% | -7.2 [-10.2, -4.2] |
| D2b/gsm8k | 1302 | 58.8% | 56.9% | -1.9 [-4.8, +0.9] |
| D3/gsm8k | 872 | 87.6% | 84.7% | -2.9 [-5.0, -0.7] |
| D1/ifeval | 385 | 72.1% | 55.7% | -16.4 [-20.6, -12.1] |
| D2a/ifeval | 540 | 47.4% | 42.5% | -4.9 [-7.9, -1.9] |
| D2b/ifeval | 533 | 52.9% | 49.8% | -3.1 [-5.6, -0.6] |
| D3/ifeval | 412 | 69.9% | 64.8% | -5.1 [-8.6, -1.7] |

## Budget curve: correct and finished within $L$ generated tokens

$\mathrm{Acc}(L) = P(\text{correct}, \text{finished}, N \le L)$, $N$ = generated tokens. "Acc" is the run's accuracy, which also counts cut answers graded correct.

| Config / benchmark | Model | $L = 4096$ | $L = 8192$ | $L = 16384$ | $L = 32768$ | Acc |
|---|---|---|---|---|---|---|
| D0/math500 | base | 3.6% | 6.4% | 7.4% | 7.4% | 8.2% |
| D0/math500 | p50c | 6.8% | 13.2% | 29.4% | 51.0% | 51.0% |
| D0/math500 | $\Delta$ (pp) | +3.2 [+1.0, +5.6] | +6.8 [+4.2, +9.6] | +22.0 [+18.2, +25.8] | +43.6 [+39.2, +47.8] | |
| D1/math500 | base | 1.7% | 4.1% | 10.1% | 18.6% | 19.1% |
| D1/math500 | p50c | 3.4% | 7.3% | 17.0% | 27.1% | 27.2% |
| D1/math500 | $\Delta$ (pp) | +1.7 [+0.7, +2.8] | +3.2 [+1.7, +4.8] | +6.9 [+4.8, +9.1] | +8.5 [+6.0, +11.0] | |
| D2a/math500 | base | 24.6% | 32.7% | 42.9% | 46.0% | 46.0% |
| D2a/math500 | p50c | 21.7% | 30.0% | 40.1% | 41.8% | 41.8% |
| D2a/math500 | $\Delta$ (pp) | -2.9 [-5.5, -0.3] | -2.7 [-5.5, +0.2] | -2.8 [-6.1, +0.4] | -4.2 [-7.3, -1.1] | |
| D2b/math500 | base | 41.1% | 45.4% | 48.5% | 49.3% | 49.5% |
| D2b/math500 | p50c | 38.3% | 43.7% | 49.4% | 51.8% | 51.8% |
| D2b/math500 | $\Delta$ (pp) | -2.8 [-5.6, +0.1] | -1.7 [-4.8, +1.4] | +0.9 [-2.3, +4.2] | +2.5 [-0.8, +5.9] | |
| D3/math500 | base | 2.0% | 15.1% | 34.8% | 38.2% | 38.2% |
| D3/math500 | p50c | 7.6% | 13.9% | 29.6% | 54.6% | 54.9% |
| D3/math500 | $\Delta$ (pp) | +5.6 [+3.9, +7.4] | -1.2 [-3.8, +1.4] | -5.2 [-8.1, -2.4] | +16.4 [+13.5, +19.4] | |
| D1/gsm8k | base | 3.9% | 9.9% | 21.4% | 33.6% | 33.8% |
| D1/gsm8k | p50c | 6.4% | 17.7% | 33.4% | 47.2% | 47.5% |
| D1/gsm8k | $\Delta$ (pp) | +2.6 [+1.2, +4.0] | +7.8 [+5.8, +9.9] | +12.1 [+9.7, +14.4] | +13.6 [+11.1, +16.2] | |
| D2a/gsm8k | base | 38.6% | 44.3% | 48.1% | 48.3% | 48.3% |
| D2a/gsm8k | p50c | 31.2% | 38.0% | 41.1% | 41.1% | 41.1% |
| D2a/gsm8k | $\Delta$ (pp) | -7.4 [-10.2, -4.5] | -6.3 [-9.2, -3.4] | -7.1 [-10.1, -4.1] | -7.2 [-10.2, -4.2] | |
| D2b/gsm8k | base | 56.0% | 57.5% | 58.1% | 58.1% | 58.2% |
| D2b/gsm8k | p50c | 54.8% | 56.0% | 56.6% | 56.6% | 56.6% |
| D2b/gsm8k | $\Delta$ (pp) | -1.1 [-3.9, +1.5] | -1.5 [-4.3, +1.2] | -1.4 [-4.2, +1.3] | -1.4 [-4.2, +1.3] | |
| D3/gsm8k | base | 5.0% | 39.5% | 60.4% | 60.6% | 60.6% |
| D3/gsm8k | p50c | 16.5% | 36.2% | 61.5% | 67.4% | 67.6% |
| D3/gsm8k | $\Delta$ (pp) | +11.4 [+9.5, +13.5] | -3.3 [-6.2, -0.3] | +1.1 [-1.5, +3.6] | +6.8 [+4.3, +9.2] | |
| D1/ifeval | base | 34.2% | 54.4% | 57.8% | 57.9% | 58.8% |
| D1/ifeval | p50c | 43.0% | 50.9% | 52.3% | 52.5% | 52.5% |
| D1/ifeval | $\Delta$ (pp) | +8.8 [+5.2, +12.4] | -3.5 [-7.5, +0.4] | -5.5 [-9.3, -1.5] | -5.4 [-9.3, -1.4] | |
| D2a/ifeval | base | 47.0% | 47.2% | 47.3% | 47.3% | 47.3% |
| D2a/ifeval | p50c | 42.3% | 42.3% | 42.4% | 42.4% | 42.4% |
| D2a/ifeval | $\Delta$ (pp) | -4.6 [-7.6, -1.8] | -4.9 [-7.8, -2.0] | -4.9 [-7.9, -1.9] | -4.9 [-7.9, -1.9] | |
| D2b/ifeval | base | 52.1% | 52.2% | 52.3% | 52.3% | 52.7% |
| D2b/ifeval | p50c | 49.8% | 49.8% | 49.8% | 49.8% | 49.8% |
| D2b/ifeval | $\Delta$ (pp) | -2.3 [-4.9, +0.3] | -2.4 [-5.0, +0.2] | -2.5 [-5.0, +0.1] | -2.5 [-5.0, +0.1] | |
| D3/ifeval | base | 21.9% | 56.7% | 57.4% | 58.0% | 64.1% |
| D3/ifeval | p50c | 51.1% | 60.7% | 61.8% | 61.8% | 62.0% |
| D3/ifeval | $\Delta$ (pp) | +29.2 [+25.6, +32.8] | +4.1 [+0.6, +7.4] | +4.4 [+1.0, +7.8] | +3.8 [+0.5, +7.1] | |

## D3 and D0 are equal in distribution, not sample by sample

- base: of 32 D3 samples on MATH-500 with seed 0 that the breaker did not cut, 0 have the same thinking and output length as D0's seed-0 sample.
- p50c: of 318 D3 samples on MATH-500 with seed 0 that the breaker did not cut, 1 have the same thinking and output length as D0's seed-0 sample.
