# Team-form FDR → future team performance

Form at decision GW **t** (GW < t only): `team_actual[g] = Σ player total_points` that GW (**assistant managers stripped**: 0 minutes + pts > 0); **`team_mu` = expanding mean** of those GW totals. `team_se = SD/√n` kept as diagnostic only. **Locked FDR:** `fdr = team_mu_own(t) − team_mu_opp(t)` on fixtures **t+3 … t+7**. Outcome: same `team_actual` at target GW; margin = own − opp.

n sides = **10536**

## Correlations (raw-μ FDR)

- FDR vs pts margin: Pearson **+0.349** · Spearman **+0.352**
- FDR vs own pts: Pearson **+0.295** · Spearman **+0.308**
- team_mu alone vs own pts: Pearson **+0.260** · Spearman **+0.253**

## By lead (k = target − t)

| lead | n | Pearson(margin) | Spearman(margin) |
| ---: | ---: | ---: | ---: |
| 3 | 2266 | +0.342 | +0.345 |
| 4 | 2186 | +0.340 | +0.346 |
| 5 | 2108 | +0.353 | +0.357 |
| 6 | 2028 | +0.361 | +0.363 |
| 7 | 1948 | +0.349 | +0.347 |

## By season

| season | n | Pearson(margin) | Spearman(margin) |
| --- | ---: | ---: | ---: |
| 2022-23 | 2736 | +0.366 | +0.373 |
| 2023-24 | 2600 | +0.359 | +0.373 |
| 2024-25 | 2600 | +0.399 | +0.399 |
| 2025-26 | 2600 | +0.252 | +0.242 |

## Decile calibration (mean margin by FDR decile)

| decile μ(FDR) | n | mean margin | mean pts |
| ---: | ---: | ---: | ---: |
| -18.589 | 1054 | -24.52 | 35.7 |
| -11.094 | 1054 | -16.55 | 37.1 |
| -6.934 | 1053 | -10.84 | 40.9 |
| -3.809 | 1056 | -2.96 | 43.8 |
| -1.181 | 1059 | +0.32 | 45.0 |
| +1.204 | 1047 | -0.40 | 44.6 |
| +3.814 | 1052 | +3.05 | 46.8 |
| +6.935 | 1054 | +10.83 | 51.8 |
| +11.096 | 1053 | +16.56 | 53.7 |
| +18.589 | 1054 | +24.52 | 60.2 |

Plot: `data/plots/team_form_fdr_calibration.png`

JSON: `data/processed/team_form_fdr_summary.json` · pairs: `data/processed/team_form_fdr_pairs.csv`
