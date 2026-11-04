# RQ1 results

Registered analysis of `prereg/criteria.md`, computed by `python -m wsprev.rq1` on 2026-11-04 19:27 (git 27ab278). D3 threshold $\tau = 0.7$.

## Edit effect on MATH-500 (500 problems; 95% bootstrap CIs, 10,000 resamples)

| Configuration | base | p50c | $\Delta$ (pp) | 95% CI (pp) |
|---|---|---|---|---|
| D0 |   8.2% |  51.0% | 42.8 | [38.6, 47.0] |
| D1 |  19.1% |  27.2% | 8.1 | [5.6, 10.7] |
| D2b |  49.5% |  51.8% | 2.3 | [-1.0, 5.6] |
| D3 |  38.2% |  54.9% | 16.7 | [13.8, 19.7] |

## Hypotheses (margin $m = 0.05$, Holm over H2–H4 at $\alpha = 0.05$)

| | Statement | Estimate (pp) | $p$ | $p$ (Holm) | Supported |
|---|---|---|---|---|---|
| H1 | $\Delta_{\mathrm{D0}} \ge 0.30$ | 42.8 | | | yes |
| H2 | $\Delta_{\mathrm{D1}} \lt \Delta_{\mathrm{D0}}$ | 34.7 | 0.0000 | 0.0000 | yes |
| H3 | $\Delta_{\mathrm{D3}} \lt m$ | 16.7 | 1.0000 | 1.0000 | no |
| H4 | $\Delta_{\mathrm{D2b}} \lt m$ | 2.3 | 0.0555 | 0.1110 | no |

## Conclusion

The edit retains a substantial gain under the fair baselines we tested.

## Cross comparison

$\operatorname{Acc}(\mathrm{p50c}, \mathrm{D0}) - \operatorname{Acc}(\mathrm{base}, \mathrm{D3}) = 12.8$ pp (95% CI [9.3, 16.3]); D3 was chosen on the practice SELECT split.

## Edit effect on every benchmark (secondary, no tests)

| Benchmark | Configuration | Problems | base | p50c | $\Delta$ (pp) | 95% CI (pp) | Data |
|---|---|---|---|---|---|---|---|
| math500 | D0 | 500 |   8.2% |  51.0% | 42.8 | [38.6, 47.0] | this study |
| math500 | D1 | 500 |  19.1% |  27.2% | 8.1 | [5.6, 10.7] | this study |
| math500 | D2a | 500 |  46.0% |  41.8% | -4.2 | [-7.3, -1.1] | this study |
| math500 | D2b | 500 |  49.5% |  51.8% | 2.3 | [-1.0, 5.6] | this study |
| math500 | D3 | 500 |  38.2% |  54.9% | 16.7 | [13.8, 19.7] | this study |
| gsm8k | D0 | 1319 |  13.8% |  67.1% | 53.3 | [51.1, 55.5] | original evaluation |
| gsm8k | D1 | 1319 |  33.8% |  47.5% | 13.6 | [11.0, 16.4] | this study |
| gsm8k | D2a | 1319 |  48.3% |  41.1% | -7.2 | [-10.2, -4.2] | this study |
| gsm8k | D2b | 1319 |  58.2% |  56.6% | -1.5 | [-4.3, 1.3] | this study |
| gsm8k | D3 | 1319 |  60.6% |  67.6% | 7.0 | [4.5, 9.4] | this study |
| ifeval | D0 | 541 |  21.0% |  62.3% | 41.3 | [37.4, 45.2] | original evaluation |
| ifeval | D1 | 541 |  58.8% |  52.5% | -6.3 | [-10.2, -2.4] | this study |
| ifeval | D2a | 541 |  47.3% |  42.4% | -4.9 | [-7.9, -1.9] | this study |
| ifeval | D2b | 541 |  52.7% |  49.8% | -2.9 | [-5.4, -0.4] | this study |
| ifeval | D3 | 541 |  64.1% |  62.0% | -2.1 | [-5.4, 1.1] | this study |

## Every run

| Model | Config | Benchmark | Problems | Samples | Accuracy | Cut | Mean tokens | Acc. finished | Acc. cut | Breaker |
|---|---|---|---|---|---|---|---|---|---|---|
| base | D0 | math500 | 500 | 500 |   8.2% |  92.6% | 30,721 | 100.0% |   0.9% |   0.0% |
| base | D1 | gsm8k | 1319 | 1319 |  33.8% |  64.1% | 26,161 |  93.7% |   0.4% |   0.0% |
| base | D1 | ifeval | 541 | 1082 |  58.8% |  18.5% | 9,630 |  71.0% |   5.0% |   0.0% |
| base | D1 | math500 | 500 | 1000 |  19.1% |  81.0% | 29,572 |  97.9% |   0.6% |   0.0% |
| base | D2a | gsm8k | 1319 | 1319 |  48.3% |   0.0% | 4,150 |  48.3% |   n/a |   0.0% |
| base | D2a | ifeval | 541 | 1082 |  47.3% |   0.0% | 325 |  47.3% |   n/a |   0.0% |
| base | D2a | math500 | 500 | 1000 |  46.0% |   0.1% | 8,814 |  46.0% |   0.0% |   0.0% |
| base | D2b | gsm8k | 1319 | 1319 |  58.2% |   1.3% | 1,552 |  58.8% |   5.9% |   0.0% |
| base | D2b | ifeval | 541 | 1082 |  52.7% |   0.8% | 541 |  52.7% |  44.4% |   0.0% |
| base | D2b | math500 | 500 | 1000 |  49.5% |  15.0% | 8,467 |  58.0% |   1.3% |   0.0% |
| base | D3 | gsm8k | 1319 | 1319 |  60.6% |  30.8% | 7,801 |  87.5% |   0.0% |  87.9% |
| base | D3 | ifeval | 541 | 1082 |  64.1% |  15.6% | 5,215 |  68.8% |  39.1% |  74.1% |
| base | D3 | math500 | 500 | 1000 |  38.2% |  60.0% | 10,866 |  95.5% |   0.0% |  93.1% |
| p50c | D0 | math500 | 500 | 500 |  51.0% |  29.8% | 21,453 |  72.6% |   0.0% |   0.0% |
| p50c | D1 | gsm8k | 1319 | 1319 |  47.5% |  35.6% | 21,258 |  73.4% |   0.6% |   0.0% |
| p50c | D1 | ifeval | 541 | 1082 |  52.5% |   1.2% | 3,982 |  53.1% |   0.0% |   0.0% |
| p50c | D1 | math500 | 500 | 1000 |  27.2% |  66.9% | 27,145 |  81.9% |   0.1% |   0.0% |
| p50c | D2a | gsm8k | 1319 | 1319 |  41.1% |   0.0% | 4,176 |  41.1% |   n/a |   0.0% |
| p50c | D2a | ifeval | 541 | 1082 |  42.4% |   0.1% | 305 |  42.5% |   0.0% |   0.0% |
| p50c | D2a | math500 | 500 | 1000 |  41.8% |   0.0% | 8,363 |  41.8% |   n/a |   0.0% |
| p50c | D2b | gsm8k | 1319 | 1319 |  56.6% |   0.0% | 1,265 |  56.6% |   n/a |   0.0% |
| p50c | D2b | ifeval | 541 | 1082 |  49.8% |   0.0% | 252 |  49.8% |   n/a |   0.0% |
| p50c | D2b | math500 | 500 | 1000 |  51.8% |   0.5% | 6,175 |  52.1% |   0.0% |   0.0% |
| p50c | D3 | gsm8k | 1319 | 1319 |  67.6% |  10.1% | 10,427 |  75.0% |   1.5% |  12.5% |
| p50c | D3 | ifeval | 541 | 1082 |  62.0% |   1.0% | 3,664 |  62.5% |  18.2% |   2.9% |
| p50c | D3 | math500 | 500 | 1000 |  54.9% |  33.7% | 18,679 |  82.4% |   0.9% |  35.1% |

## Problems where neither model was cut

| Config / benchmark | Problems | base | p50c |
|---|---|---|---|
| D0/math500 | 37 | 100.0% | 100.0% |
| D1/gsm8k | 413 |  95.6% |  88.6% |
| D1/ifeval | 385 |  72.1% |  55.7% |
| D1/math500 | 55 | 100.0% |  99.1% |
| D2a/gsm8k | 1319 |  48.3% |  41.1% |
| D2a/ifeval | 540 |  47.4% |  42.5% |
| D2a/math500 | 499 |  46.1% |  41.9% |
| D2b/gsm8k | 1302 |  58.8% |  56.9% |
| D2b/ifeval | 533 |  52.9% |  49.8% |
| D2b/math500 | 375 |  61.9% |  59.6% |
| D3/gsm8k | 872 |  87.6% |  84.7% |
| D3/ifeval | 412 |  69.9% |  64.8% |
| D3/math500 | 152 |  97.4% |  98.0% |
