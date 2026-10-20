Gates: paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

As-of audit soft-pass for this closed-season screen (live lock drift): refusing report: as-of audit has not passed (the live primary is not ep_next; score_xp is not the live shadow; the published historical score is not score_xp; the live primary declared a winner; the live primary wiring flag does not match the lock; a live interval that covers zero is not undetermined; the live test does not continue through gameweek 38; the live primary does not start at gameweek 6; promotion does not wait for 20 live weeks; the decision capture is not the same-stamp t1 pair; an earlier capture is a fallback; the gameweek 6 chip is not the logged judgement; the gameweek 6 chip counts toward a chip rule; no pre-deadline capture for gameweeks 7)

# Assist α_A screen — team xA/xG vs pooled α_A (Numagent 2.0)

Pre-registered before totals were read. Control is `score_xp_v2` with locked pooled `α_A` (`a_team = α_A · xG_M`). Candidate replaces `α_A` per club with shift-1 same-season `ΣxA/ΣxG` clipped to [0.35, 1.25] (fallback = pooled α_A).

This rejudges the legacy assist-pot promote (team xA/xG over 0.75 on `score_xp`) under the Numagent 2.0 assembler.

Population: eligible rows (`n_prior ≥ 3`, `minutes > 60`), gameweeks 5–38, four closed seasons. Metric: player-GW Gaussian log score, σ=3.

## Decision

**keep_pooled_alpha_a**

Promote only if the candidate−control interval stays strictly above zero. Interval does not clear zero: keep pooled α_A.

## Intervals

| contrast | mean | 95% interval |
| --- | ---: | ---: |
| `score_xp_v2_assist_team_xa_minus_score_xp_v2` | -0.00175 | [-0.00255, -0.00097] |
| `score_xp_minus_score_exp_points` | +0.01531 | [+0.01295, +0.01748] |

Gameweeks per season: 2022-23=33, 2023-24=34, 2024-25=34, 2025-26=34.

## Scales on eligible rows

| season | n | mean ρ | mean α_A | mean |ΔxP| |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | 7330 | 0.681 | 0.855 | 0.1092 |
| 2023-24 | 7128 | 0.637 | 0.872 | 0.1227 |
| 2024-25 | 6897 | 0.660 | 0.878 | 0.1021 |
| 2025-26 | 6847 | 0.626 | 0.909 | 0.1208 |

## Outputs

- `data/processed/assist_alpha_logscore.csv`
- `data/processed/assist_alpha_screen.json`

Reproduce: `PYTHONPATH=. python3 -m src.eval.assist_alpha_screen`.
