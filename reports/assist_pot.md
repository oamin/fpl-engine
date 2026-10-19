Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Assist pot screen — team xA/xG vs fixed ρ=0.75

Pre-registered before totals were read. Control is published `score_xp` (`lam_assist = 0.75 · lam_scored`). Candidate replaces ρ with shift-1 `exp_team_xa / exp_team_xg` (clipped), then recomputes `compute_xp`.

Population: eligible rows (`n_prior ≥ 3`, `xmi ≥ 45`), gameweeks 5–38, four closed seasons. Metric: player-GW Gaussian log score, σ=3. Bootstrap resamples gameweeks inside each season.

## Decision

**promote_team_xa_xg**

Promote only if the candidate−control interval stays strictly above zero. Interval clears zero: switch the published assist pot to `team_xa_xg`.

## Intervals

| contrast | mean | 95% interval |
| --- | ---: | ---: |
| `score_xp_assist_team_xa_minus_score_xp` | +0.00041 | [+0.00013, +0.00069] |
| `score_xp_minus_score_exp_points` | +0.02597 | [+0.01973, +0.03226] |

Gameweeks per season (assist contrast): 2022-23=33, 2023-24=34, 2024-25=34, 2025-26=34.

## Candidate ρ on eligible rows

| season | n | mean ρ | p10 | p90 | mean |ΔxP| |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 7686 | 0.659 | 0.523 | 0.750 | 0.0297 |
| 2023-24 | 7692 | 0.636 | 0.528 | 0.767 | 0.0359 |
| 2024-25 | 7718 | 0.663 | 0.570 | 0.776 | 0.0327 |
| 2025-26 | 7658 | 0.626 | 0.516 | 0.725 | 0.0361 |

## Outputs

- `data/processed/assist_pot_logscore.csv`
- `data/processed/assist_pot_screen.json`

Reproduce: `python -m src.eval.assist_pot`.
