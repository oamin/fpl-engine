Gates: paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

As-of audit soft-pass for this closed-season screen (live lock drift): refusing report: as-of audit has not passed (live_primary Numagent 2.0 lock drift / missing GW7 capture)

# Item 4 — Player G/A finishing screen (Numagent 2.0)

Pre-registered before totals were read. Control is `score_xp_v2` (rates → shares → `score_team_side` on FD AH+OU pots). Candidate multiplies `rG`/`rA` by player shift-1 finishing `(ΣG+κ)/(ΣxG+κ)` and the assist analogue (κ=2, clipped to [0.65, 1.40]) before shares.

Population: eligible rows (`n_prior ≥ 3`, `minutes > 60`), gameweeks 5–38, four closed seasons. Metric: player-GW Gaussian log score, σ=3. Bootstrap resamples gameweeks inside each season.

## Decision

**keep_share_times_lambda**

Promote only if the candidate−control interval stays strictly above zero. Interval does not clear zero: keep share×λ rates.

## Intervals

| contrast | mean | 95% interval |
| --- | ---: | ---: |
| `score_xp_v2_finish_minus_score_xp_v2` | -0.00187 | [-0.00256, -0.00117] |
| `score_xp_minus_score_exp_points` | +0.01531 | [+0.01295, +0.01748] |

Gameweeks per season (finish contrast): 2022-23=33, 2023-24=34, 2024-25=34, 2025-26=34.

## Finish scales on eligible rows

| season | n | mean finish_g | mean finish_a | p10_g | p90_g | mean |ΔxP| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 7330 | 1.130 | 1.174 | 0.867 | 1.400 | 0.0802 |
| 2023-24 | 7128 | 0.999 | 1.070 | 0.727 | 1.356 | 0.0685 |
| 2024-25 | 6897 | 0.985 | 1.062 | 0.709 | 1.316 | 0.0542 |
| 2025-26 | 6847 | 0.989 | 1.075 | 0.719 | 1.333 | 0.0534 |

## Outputs

- `data/processed/player_ga_logscore.csv`
- `data/processed/player_ga_screen.json`

Reproduce: `PYTHONPATH=. python3 -m src.eval.player_ga_screen`.
