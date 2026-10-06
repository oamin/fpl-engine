Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Procedure audit

One pre-registered batch. The gate is the player-GW Gaussian log score with σ = 3, fixed before the totals were read. The cluster is a gameweek. Gameweeks are resampled inside each season, then pooled. B = 1000, seed 0, 95% interval. Gameweeks 5–38.

## Likelihood

A positive mean is a higher log score for the first column.

| comparison | mean [95% interval] | player-GW rows |
|---|---:|---:|
| score_xp − score_exp_points | +0.0153 [+0.0130, +0.0175] | 102315 |
| score_xp − score_official_xp | -0.0016 [-0.0163, +0.0125] | 102315 |
| score_official_xp − score_exp_points | +0.0169 [+0.0035, +0.0306] | 102315 |

The same delta by season. Each cell is the mean of that season's gameweek deltas.

| season | score_xp − exp | score_xp − official | official − exp |
|---|---:|---:|---:|
| 2022-23 | +0.0084 | -0.0417 | +0.0501 |
| 2023-24 | +0.0171 | -0.0506 | +0.0677 |
| 2024-25 | +0.0157 | -0.0357 | +0.0513 |
| 2025-26 | +0.0198 | +0.1203 | -0.1005 |

2022-23 has no sheet rows in gameweek 7, so that week is absent. 33 gameweeks remain.

On the pooled gameweeks, score_xp minus score_exp_points is +0.0153 [+0.0130, +0.0175], and the interval stays above zero. score_xp minus score_official_xp is -0.0016 [-0.0163, +0.0125], and the interval covers zero. The season means of that second comparison are 2022-23 -0.0417, 2023-24 -0.0506, 2024-25 -0.0357, 2025-26 +0.1203. score_official_xp minus score_exp_points is +0.0169 [+0.0035, +0.0306], and the interval stays above zero. Its season means are 2022-23 +0.0501, 2023-24 +0.0677, 2024-25 +0.0513, 2025-26 -0.1005.

The likelihood is the mean over every finite player-GW row, including 0-minute rows. It is not the eleven.

## Stripped XI, descriptive

Eligible rows only: at least three prior sheet rows and expected minutes at least 45, both from shift-1 history that includes 0-minute weeks. The total is the eleven's points plus the highest points inside that eleven. That captain is the realised maximum, so the column is a sanity check. A week one score cannot fill is dropped from all three. This table is not a pass.

| season | weeks | score_xp | score_exp_points | score_official_xp |
|---|---:|---:|---:|---:|
| 2022-23 | 33 | 2312 | 2530 | 3464 |
| 2023-24 | 34 | 2450 | 2389 | 3575 |
| 2024-25 | 34 | 2614 | 2285 | 3251 |
| 2025-26 | 34 | 2259 | 2071 | 1992 |

There is no budget and no transfer constraint, and the captain is the highest realised score in the eleven. Official xP has the highest total in 2022-23, 2023-24, 2024-25. score_xp has the highest total in 2025-26. These totals are not a pass.

## What was wrong, and what changed

The buy pool had been the players who played the gameweek being scored. `load_player_logs` dropped `minutes <= 0`, and `build_frames` did the same before the priors. A purchase was a player who had already played. A sheet row now stays whether the minutes are 0 or not. A row that misses its fixture stays, with an empty market cell. Expected minutes, expected points, the three-week roll, expected xG, expected xA, and the defensive-contribution rate stay shift-1, and the shift includes the 0-minute weeks. A missing player prior is not filled from that season's position mean. The row is not eligible. A team with no shifted history takes xG 1.40 and xA 1.05. Those two numbers are fixed. They are not a mean of the season being scored. Eligible means at least three prior sheet rows and expected minutes at least 45. The current week's minutes, points, and the fact that a positive-minute row exists do not enter that flag. A 0-minute row with a high prior is eligible and scores 0 if it is picked. A player who appears for the first time in the scored week is not eligible.

Closing odds had been preferred to opening odds. `AvgCH` of 1.2 with `AvgH` of 3.0 now uses 3.0. The 1X2 group is Avg, then B365, then Pinnacle. The 2.5 total is Avg, then B365, and never `AvgC>2.5`. The asian line is `AHh`, with opening asian prices. A fixture whose opening 1X2 is missing is dropped even when the closing price is present.

The season-total bar of +34 is retired. One season's paired noise was larger than that bar, and every finished season had already been used as a screen. The interval above is the comparison. A climb total is not a winner. `experiments/matrix.json` no longer carries `pass_margin`, and `search_protocol` does not label a season total as a winner.

Official xP is the Vaastav `xP` column on the same row, stored as `official_xp` and scored as `score_official_xp`. It is that week's pre-deadline forecast. It is not shifted, and it is not an input to `compute_xp`. `total_points` stays the outcome. `exp_points` remains the expanding mean of past points. It is a different baseline.

Stage 13's information-coefficient gate failed: xP trailed expected points at horizon 8. Stage 18's ridge tied xP under the budget, a difference of −2. Both were set aside because the stripped climb was treated as the gate. That climb is the comparison most exposed to the played-only pool. They stay set aside. This batch does not reopen them as a pass.

The goalkeeper goal inside `compute_xp` now reads `GOAL_POINTS` from `src/rules/fpl_2026.py`, so a goalkeeper goal is 6. Historical `total_points` are not rescaled. The default formation list is `OFFICIAL_FORMATIONS`, which includes 5-2-3. A tie keeps the earlier shape. The harness in `src/models/` was left in place. The leak is closed at `load_player_logs`, the priors, and `build_frames`, and a result has one writer, `write_gated_report`. Pytest and ruff are declared. The workflow runs the gate tests. The full player-GW frame is not committed. The committed evidence is the gameweek log-score table.

Historical sheets have no chance of playing. Availability on this comparison is the minutes history, including zeros. Live news tags stay on the live path. No historical news tag was invented.

## Rules now in force

A result is not reported because a review sentence says so. The as-of audit has to pass, and the paired intervals have to exist for all four closed seasons. `write_gated_report` raises otherwise. Gemini may still be asked about a formula. That exchange is not the certificate.

The next batch is one pre-registered comparison on this protocol, judged on the player-GW likelihood. Objective-function search (Sharpe, ownership, churn) stays parked. A captain model and a chip simulator stay deferred. A climb is a sanity check.

The live 2026/27 holdout is frozen. `data/live/HOLDOUT_FREEZE.json` holds the hashes. Those snapshots were not rewritten, and no parameter was fit on that season. No Odds API call was made. This comparison does not include 2026-27.

The published Gameweeks 1–5 total of 280 used the played-only pool and the shorter formation list. It was not recomputed on the repaired pool, and it is not restated here as a new measurement.

- `data/processed/player_gw_logscore.csv`
- `data/processed/player_gw_xi_sanity.csv`
- `experiments/protocol.json`
