Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Procedure audit

One pre-registered batch. The gate is the player-GW Gaussian log score with σ = 3, fixed before the totals were read. The cluster is a gameweek. Gameweeks are resampled inside each season, then pooled. Player rows inside a gameweek are not resampled. B = 1000, seed 0, 95% interval. Gameweeks 5–38. The certified comparison is score_xp against the shift-1 mean of points. Scraped official xP is not in this table.

## Likelihood

A positive mean is a higher log score for the first column.

| comparison | mean [95% interval] | player-GW rows |
|---|---:|---:|
| score_xp − score_exp_points | +0.0153 [+0.0130, +0.0175] | 102315 |

The same delta by season. Each cell is the mean of that season's gameweek deltas.

| season | score_xp − exp |
|---|---:|
| 2022-23 | +0.0084 |
| 2023-24 | +0.0171 |
| 2024-25 | +0.0157 |
| 2025-26 | +0.0198 |

2022-23 has no sheet rows in gameweek 7 on the expected-points comparison, so that week is absent. 33 gameweeks remain.

On the pooled gameweeks, score_xp minus score_exp_points is +0.0153 [+0.0130, +0.0175], and the interval stays above zero.

The likelihood is the mean over every finite player-GW row, including 0-minute rows. A Gaussian log score with σ fixed at 3 is mean squared error up to a constant, so non-starters dominate it. The rank slice below is the decision-relevant cut. It is not the eleven.

## Scraped xP, withdrawn

The Vaastav README says `xP` comes from the FPL field `ep_this`, and the scraper runs after each gameweek ends. If FPL updates `ep_this` after the matches, the scraped value contains information that was not available before the deadline. The maintainer says that cadence is not documented. Weekly updates of that repository stopped after 2024-25. A season now has three updates. `modified` is not a scrape clock. These sheets have no per-row capture time, so they fail the pre-deadline check. They are not a benchmark and not a feature. The figures below are the last run that used them. They are not a result, and they are not a reason to stop work on the forecast.

Last scraped log score, not certified: score_xp minus official xP −0.0565 [−0.0680, −0.0442] on the three seasons with at least 20 filled weeks. Spearman on eligible rows −0.3487 [−0.3789, −0.3188]. Top-15 intersection +0.2174 [−0.0257, +0.4949]. Walk-forward coefficient on score_xp −0.0245 [−0.0356, −0.0139]. The −0.35 rank gap is unexplained. It is not evidence that the scrape is post-match, and it is not evidence that it is clean.

Filled weeks in the cache, maximum xP greater than 0, are listed so the pattern can be checked. An all-zero week is an unfilled scrape. That description is not a score.

2022-23: unfilled gameweeks 12, 36. `modified` is absent. Blanks with xP at least 4: 96. value median 45.0, max 131.0. selected mean 235503. transfers_balance is not identically 0.

2023-24: unfilled gameweeks 26. `modified` is absent. Blanks with xP at least 4: 86. value median 45.0, max 145.0. selected mean 201081. transfers_balance is not identically 0.

2024-25: unfilled gameweeks 22, 32, 34. `modified` is boolean false on every row. Blanks with xP at least 4: 251. value median 45.0, max 154.0. selected mean 227979. transfers_balance is not identically 0.

2025-26: unfilled gameweeks 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37. `modified` is boolean false on every row. Blanks with xP at least 4: 14. value median 46.0, max 151.0. selected mean 241112. transfers_balance is not identically 0.

`modified` by season: 2022-23: absent; 2023-24: absent; 2024-25: boolean false on every row; 2025-26: boolean false on every row. It is not a scrape clock. No historical news time was invented. The README warning is the provenance.

`value` is the gameweek price in tenths. `selected` is ownership. `transfers_balance` is the transfer column on the same sheet. None of the three enters the sum inside `compute_xp`. `value` is copied to `baseline_value` and is not the engine score. A filled week still contains blanks whose official xP is at least 4, so a filled column is not a rewrite of the points.

## Stripped XI, descriptive

Eligible rows only: at least three prior sheet rows and expected minutes at least 45, both from shift-1 history that includes 0-minute weeks. The total is the eleven's points plus the highest points inside that eleven. That captain is the realised maximum, so the column is a sanity check. A week one score cannot fill is dropped from both columns. This table is not a pass.

| season | weeks | score_xp | score_exp_points |
|---|---:|---:|---:|
| 2022-23 | 33 | 2312 | 2530 |
| 2023-24 | 34 | 2450 | 2389 |
| 2024-25 | 34 | 2614 | 2285 |
| 2025-26 | 34 | 2259 | 2071 |

There is no budget and no transfer constraint, and the captain is the highest realised score in the eleven. These totals are not a pass.

## Decision slice

Eligible rows only. Within a season, gameweek, and position, the rank delta is Spearman(score, points) for the first column minus the same rank for the second. A position with fewer than three finite rows, or a constant score, is skipped. The top 15 by each score are paired on the players both lists name, when that intersection has at least five players and the position has at least fifteen eligible rows. The top-15 delta is the paired Gaussian log score on that intersection. A gameweek delta is the mean across positions. The bootstrap still resamples gameweeks.

| slice | comparison | mean [95% interval] | pooled gameweeks | left out |
|---|---|---:|---|---|
| spearman | score_xp − score_exp_points | +0.0210 [+0.0017, +0.0413] | 2022-23 33, 2023-24 34, 2024-25 34, 2025-26 34 | — |
| top15 | score_xp − score_exp_points | +0.0249 [+0.0043, +0.0434] | 2022-23 33, 2023-24 34, 2024-25 34, 2025-26 34 | — |

On the rank slice, score_xp minus score_exp_points is +0.0210 [+0.0017, +0.0413], and the interval stays above zero. Scraped official xP is not in this slice.

## Live encompassing test

The formula is unchanged: points on an intercept, official xP, and score_xp. Both inputs have to be pre-deadline captures, and the points have to be the played gameweek. Eligible weeks start at 2026-27 gameweek 6. A survival call needs 20 such gameweeks. This batch does not make that call.

Status: held. Evaluable gameweeks: 0. Outcomes in the 2026-27 log run through gameweek 5. Engine captures: data/predictions/2026-27/gw06/20261007T070450Z.csv (git 2026-10-07T07:06:17+00:00). Official captures: data/predictions/2026-27/gw06/official_20261007T080406Z.csv. No gameweek has a pre-deadline official xP, a pre-deadline engine score, and a played outcome. The stop-forecasting decision stays held. score_xp stays the forecast.

## What was wrong, and what changed

The buy pool had been the players who played the gameweek being scored. `load_player_logs` dropped `minutes <= 0`, and `build_frames` did the same before the priors. A purchase was a player who had already played. A sheet row now stays whether the minutes are 0 or not. A row that misses its fixture stays, with an empty market cell. Expected minutes, expected points, the three-week roll, expected xG, expected xA, and the defensive-contribution rate stay shift-1, and the shift includes the 0-minute weeks. A missing player prior is not filled from that season's position mean. The row is not eligible. A team with no shifted history takes xG 1.40 and xA 1.05. Those two numbers are fixed. They are not a mean of the season being scored. Eligible means at least three prior sheet rows and expected minutes at least 45. The current week's minutes, points, and the fact that a positive-minute row exists do not enter that flag. A 0-minute row with a high prior is eligible and scores 0 if it is picked. A player who appears for the first time in the scored week is not eligible.

Closing odds had been preferred to opening odds. `AvgCH` of 1.2 with `AvgH` of 3.0 now uses 3.0. The 1X2 group is Avg, then B365, then Pinnacle. The 2.5 total is Avg, then B365, and never `AvgC>2.5`. The asian line is `AHh`, with opening asian prices. A fixture whose opening 1X2 is missing is dropped even when the closing price is present.

The season-total bar of +34 is retired. One season's paired noise was larger than that bar, and every finished season had already been used as a screen. The interval above is the comparison. A climb total is not a winner. `experiments/matrix.json` no longer carries `pass_margin`, and `search_protocol` does not label a season total as a winner.

Vaastav `xP` is stored as `official_xp` and is not an input to `compute_xp`. It is not a pre-deadline forecast. The README says the scraper runs after the gameweek. `total_points` stays the outcome. `exp_points` remains the expanding mean of past points.

Stage 13's information-coefficient gate failed: xP trailed expected points at horizon 8. Stage 18's ridge tied xP under the budget, a difference of −2. Both were set aside because the stripped climb was treated as the gate. That climb is the comparison most exposed to the played-only pool. They stay set aside. This batch does not reopen them as a pass.

The goalkeeper goal inside `compute_xp` now reads `GOAL_POINTS` from `src/rules/fpl_2026.py`, so a goalkeeper goal is 6. Historical `total_points` are not rescaled. The default formation list is `OFFICIAL_FORMATIONS`, which includes 5-2-3. A tie keeps the earlier shape. The harness in `src/models/` was left in place. The leak is closed at `load_player_logs`, the priors, and `build_frames`, and a result has one writer, `write_gated_report`. Pytest and ruff are declared. The workflow runs the gate tests. The full player-GW frame is not committed. The committed evidence is the gameweek log-score table.

Historical sheets have no chance of playing. Availability on this comparison is the minutes history, including zeros. Live news tags stay on the live path. No historical news tag was invented.

## Rules now in force

A result is not reported because a review sentence says so. The as-of audit has to pass, and each closed season is either inside a paired interval or marked incomplete with fewer than 20 gameweeks. `write_gated_report` raises otherwise. This runner also refuses the file when the rank slice or the encompassing coefficient is missing. Gemini may still be asked about a formula. That exchange is not the certificate.

Stages 14–34, and the later climbs on the same played-only pool, are superseded. Their totals stay in `reports/` as the record of those questions. They are not the current comparison. See `reports/SUPERSEDED.md`. The transfer climb was not rerun. The published Gameweeks 1–5 total of 280 used the played-only pool and the shorter formation list. It was not recomputed, and it is not restated here as a new measurement.

Pull requests 53, 54, 55, and 56 stay open as finished counts on the stack. They are not extended. A new chip simulator stays deferred. They are not merged ahead of these gates. This checkout cannot merge the stack onto `main`. That merge is a PI action on GitHub. The default comparison is this gated path.

The live 2026/27 holdout is frozen. Gameweeks 1–5 of that season were already read by the chip-hurdle and horizon trials, so they are not a clean holdout. The clean holdout starts at gameweek 6. `data/live/HOLDOUT_FREEZE.json` holds the snapshot hashes. A hash of a file that keeps growing does not freeze a future week. Pre-deadline forecasts are timestamped files under `data/predictions/`, and an existing timestamp file is not overwritten. Those snapshots were not rewritten, and no parameter was fit on 2026-27. No Odds API call was made. This comparison does not include 2026-27.

- `data/processed/player_gw_logscore.csv`
- `data/processed/player_gw_slices.csv`
- `data/processed/player_gw_xi_sanity.csv`
- `experiments/protocol.json`
