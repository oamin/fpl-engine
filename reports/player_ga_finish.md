Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Item 4 — Player G/A finishing screen

Pre-registered before totals were read. Control is published `score_xp` (`e = share · λ`, FWD position goal scale on). Candidate multiplies by player shift-1 finishing `(ΣG+κ)/(ΣxG+κ)` and the assist analogue (κ=2, clipped to [0.65, 1.40]); FWD position scale off.

Population: eligible rows (`n_prior ≥ 3`, `xmi ≥ 45`), gameweeks 5–38, four closed seasons. Metric: player-GW Gaussian log score, σ=3. Bootstrap resamples gameweeks inside each season.

## Decision

**keep_share_times_lambda**

Promote only if the candidate−control interval stays strictly above zero. Interval does not clear zero: keep share×λ.

## Intervals

| contrast | mean | 95% interval |
| --- | ---: | ---: |
| `score_xp_finish_minus_score_xp` | +0.00079 | [-0.00050, +0.00197] |
| `score_xp_minus_score_exp_points` | +0.02597 | [+0.01973, +0.03226] |

Gameweeks per season (finish contrast): 2022-23=33, 2023-24=34, 2024-25=34, 2025-26=34.

## Finish scales on eligible rows

| season | n | mean finish_g | mean finish_a | p10_g | p90_g | mean |ΔxP| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 7686 | 1.131 | 1.171 | 0.868 | 1.400 | 0.1222 |
| 2023-24 | 7692 | 0.994 | 1.067 | 0.722 | 1.346 | 0.1164 |
| 2024-25 | 7718 | 0.984 | 1.062 | 0.712 | 1.316 | 0.1210 |
| 2025-26 | 7658 | 0.988 | 1.075 | 0.714 | 1.327 | 0.1107 |

## Outputs

- `data/processed/player_ga_logscore.csv`
- `data/processed/player_ga_screen.json`

Reproduce: `PYTHONPATH=. python3 -m src.eval.player_ga_screen`.
