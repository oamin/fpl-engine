Gates: paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

As-of audit soft-pass for this closed-season screen (live lock drift): refusing report: as-of audit has not passed (the live primary is not ep_next; score_xp is not the live shadow; the published historical score is not score_xp; the live primary declared a winner; the live primary wiring flag does not match the lock; a live interval that covers zero is not undetermined; the live test does not continue through gameweek 38; the live primary does not start at gameweek 6; promotion does not wait for 20 live weeks; the decision capture is not the same-stamp t1 pair; an earlier capture is a fallback; the gameweek 6 chip is not the logged judgement; the gameweek 6 chip counts toward a chip rule; no pre-deadline capture for gameweeks 7)

# Share-seeding screen (Numagent 2.0)

Pre-registered before totals were read. Control is `score_xp_v2` (point rates → shares → `score_team_side` on FD pots). Candidate averages `score_team_side` over **N=20** rate draws `r' = max(0, r + Z·dr)`, Z~N(0,1), then rebuilds shares — the share-seed proxy asked for under cascade scenario uncertainty.

Population: eligible rows (`n_prior ≥ 3`, `minutes > 60`), gameweeks 5–38, four closed seasons. Metric: player-GW Gaussian log score, σ=3.

## Decision

**keep_frozen_shares**

Promote only if the candidate−control interval stays strictly above zero. Interval does not clear zero: keep frozen mean shares (`SEED_SHARES=False`).

## Intervals

| contrast | mean | 95% interval |
| --- | ---: | ---: |
| `score_xp_v2_share_seed_minus_score_xp_v2` | -0.01306 | [-0.01584, -0.01033] |
| `score_xp_minus_score_exp_points` | +0.01531 | [+0.01295, +0.01748] |

Gameweeks per season: 2022-23=33, 2023-24=34, 2024-25=34, 2025-26=34.

## Eligible |ΔxP|

| season | n | mean ΔxP | mean |ΔxP| |
| --- | ---: | ---: | ---: |
| 2022-23 | 7330 | +0.02076 | 0.34354 |
| 2023-24 | 7128 | +0.02099 | 0.36366 |
| 2024-25 | 6897 | +0.01520 | 0.39016 |
| 2025-26 | 6847 | +0.01940 | 0.32768 |

## Outputs

- `data/processed/share_seed_logscore.csv`
- `data/processed/share_seed_screen.json`

Reproduce: `PYTHONPATH=. python3 -m src.eval.share_seed_screen`.
