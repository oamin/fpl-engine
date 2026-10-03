# Open-source predictors, and where this engine is thin

Read 2026-10-03. Nothing here was retrained, and no weights were copied. The comparison is the method, not a leaderboard on 2026/27.

## What they do

**OpenFPL** (Groos, arXiv:2508.09992, MIT) fits a separate XGBoost and random-forest ensemble for each position on public FPL and Understat data. Each input is a mean over the last 1, 3, 5, 10 and 38 matches, for the player, his club, and the opponent. Minutes come from the FPL availability tag (0, 25, 50, 75, 100), not from a private minutes file. On a prospective slice of 2024/25 it loses to FPL Review on players who do not play, and it is ahead on returns of 3 or more. The same pattern holds at one, two and three weeks ahead.

**Dastan** (SmartPlayFPL, 2026, MIT) keeps OpenFPL's features and changes the estimator. About 62% of player-gameweeks are 0 minutes, so one regression mostly learns that a player scores nothing. Their score is the chance of 60 minutes times the points given a start, plus the chance of not starting times the points given a cameo or a blank. On the same 2025/26 Gameweek 1–24 rows, Dastan beats OpenFPL on the full pool (MAE 0.948 against 1.167) and the two are tied on players who play 60 minutes. The whole gap is the blank: RMSE 0.583 against 0.964 on 0-minute rows. OpenFPL is slightly better on 3–4 point and 5-plus returns. FPL's own `ep_next` is close behind both. Adding it to Dastan moves their objective by +0.0018, inside the seed noise they measure at 0.0106. A declared set-piece order was +0.0003 and was rejected. Dastan does not pick a squad.

**fpl-solver** (Zwagerman, MIT) is a smaller public-API score: fixture difficulty, a minutes probability, and a discount for a player with no history, then an integer squad. It has no prospective table.

## Where this engine differs

The published score is a shot share times the club's goals from the 1X2 and the totals. That is a points-given-he-plays model. A week with 0 minutes is dropped before the averages, so the minutes prior is the last three appearances and a regular who is benched stays at 90. OpenFPL and Dastan both say that blank is the part a public model can actually win. On players who play, they do not separate by much, which matches a share-times-λ score being a reasonable starter model.

Two local consequences showed up on the 2026/27 squad.

A missing week was filled from the latest row in the file, including a later week. Davis and Sangaré had no score until Gameweek 4 and were still named in the earlier XI. That leak is removed. The published five-week total of 327 was that run. With past weeks only, the same rule scores 313, keeps João Pedro, and does not take a hit. The Gameweek 3 sale in the 327 record was the future score.

A scheduled-minutes column, `score_xp_sched`, puts those bench weeks into the minutes prior and leaves the share alone. On 2025/26 Gameweeks 5–38 the fast XI was −4 and the free-transfer climb was +9, with 12 hit points against 16. That is inside the +34 bar. It does not replace `score_xp`.

## What was not worth taking

Retraining the OpenFPL or Dastan ensembles would not be a clean test. Their seasons predate the 2026/27 scoring changes, and a feature effect under about 0.01 on Dastan's objective is noise. Set-piece tags and FPL's own `ep_next` did not clear that noise once a minutes model exists. The opening-odds horizon remains a separate transfer rule. On the corrected five-week run it scores 303, behind the published rule at 313.
