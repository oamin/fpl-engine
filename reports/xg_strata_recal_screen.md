Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Stratified FD centreline pot calibration (xG quartiles)

Pre-registered before totals were read (`protocol.score_xp_v2_xg_strata_recal`).

Control is `score_xp_v2` on raw FD AH+OU centreline `xg`. Candidate keeps shares, minutes, `xcs`/`xgc`, and `α_G`/`α_A`, and replaces attack pot level with a **LOSO xG-quartile multiplicative rescale**:

```
scale_k = mean(goals)_k / mean(xg)_k   # other three closed seasons
xg' = clip(xg · scale(bin(xg)), 0.05, 4.5)
```

Quartile edges come from the training fold only. Per-bin scales clipped to [0.5, 1.5]. This targets the conditional high-xG under-delivery / low-xG over-delivery in match pots — not a premium-player tax.

Population: eligible rows (`n_prior ≥ 3`, `minutes > 60`), gameweeks 5–38, four closed seasons. Metric: player-GW Gaussian log score, σ=3. Bootstrap resamples gameweeks inside each season.

## Decision

**keep_raw_fd_xg**

Promote only if the candidate−control interval stays strictly above zero. Interval does not clear zero: keep raw FD `xg`.

## Intervals

| contrast | mean | 95% interval |
| --- | ---: | ---: |
| `score_xp_v2_xg_strata_minus_score_xp_v2` | -0.00096 | [-0.00164, -0.00029] |
| `score_xp_minus_score_exp_points` | +0.01531 | [+0.01295, +0.01748] |

Gameweeks per season: 2022-23=33, 2023-24=34, 2024-25=34, 2025-26=34.

## LOSO scales (holdout ← train on other three)

| holdout | scales (low→high xG) | mean ΔxP | mean |ΔxP| | n elig |
| --- | --- | ---: | ---: | ---: |
| 2022-23 | 1.070, 0.983, 0.929, 0.979 | -0.01129 | 0.08762 | 7330 |
| 2023-24 | 1.063, 0.984, 0.923, 0.945 | -0.03987 | 0.08340 | 7128 |
| 2024-25 | 1.089, 0.984, 0.984, 0.993 | -0.00074 | 0.03407 | 6897 |
| 2025-26 | 1.064, 0.981, 0.982, 0.990 | -0.00227 | 0.03834 | 6847 |

## Secondary — side-row goals/xg (diagnostic)

Pooled across seasons. Fixed absolute xG bands for display ([0.0, 1.05, 1.35, 1.75, 5.0]). Not a promote gate.

### xg_band (fixed)

| arm | band | n | mean xG | mean goals | goals/xG |
| --- | --- | ---: | ---: | ---: | ---: |
| raw | q1_low | 609 | 0.875 | 0.934 | 1.068 |
| raw | q2 | 762 | 1.200 | 1.207 | 1.006 |
| raw | q3 | 828 | 1.537 | 1.457 | 0.948 |
| raw | q4_high | 839 | 2.179 | 2.136 | 0.980 |
| calibrated | q1_low | 448 | 0.883 | 0.882 | 0.999 |
| calibrated | q2 | 1030 | 1.196 | 1.200 | 1.004 |
| calibrated | q3 | 811 | 1.517 | 1.523 | 1.004 |
| calibrated | q4_high | 749 | 2.173 | 2.164 | 0.996 |

### ah_band

| arm | band | n | mean xG | mean goals | goals/xG |
| --- | --- | ---: | ---: | ---: | ---: |
| raw | big_dog | 738 | 0.952 | 0.965 | 1.014 |
| raw | dog | 610 | 1.240 | 1.218 | 0.982 |
| raw | even | 690 | 1.468 | 1.449 | 0.987 |
| raw | fav | 488 | 1.779 | 1.799 | 1.011 |
| raw | big_fav | 512 | 2.358 | 2.254 | 0.956 |
| calibrated | big_dog | 738 | 0.999 | 0.965 | 0.966 |
| calibrated | dog | 610 | 1.234 | 1.218 | 0.987 |
| calibrated | even | 690 | 1.423 | 1.449 | 1.019 |
| calibrated | fav | 488 | 1.715 | 1.799 | 1.049 |
| calibrated | big_fav | 512 | 2.298 | 2.254 | 0.981 |

## Outputs

- `data/processed/xg_strata_recal_logscore.csv`
- `data/processed/xg_strata_recal_side_strata.csv`
- `data/processed/xg_strata_recal_screen.json`

Reproduce: `PYTHONPATH=. python3 -m src.eval.xg_strata_recal_screen`.
