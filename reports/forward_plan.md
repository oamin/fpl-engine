# Forward plan

An outside review treated the one-gameweek note as the whole forecast. Some of that review describes the note. The formula in `compute_xp` is wider. This plan keeps the points that match the code, drops the ones that do not, and locks an order with Gemini.

A second pass on that plan added three things that change the order. The 37-point gap is split before any formula change. Calibration reports rank and error size, not only the mean. The bench's later objective is the expected value of an automatic substitute, not a minutes floor.

The gap already measured is actual points. From ojaminFC's Gameweek 1 fifteen, the past-only published path scored 313 against their 350 over Gameweeks 1–5. Five weeks are a diagnosis. They are not a verdict that the rule is worse.

## What the score already contains

`score_xp` is the sum of appearance, goals, assists, clean sheets, defensive contributions, goalkeeper saves, a small bonus proxy, minus goals conceded and a yellow-card proxy. Goals and assists are shot share times the team rate. The other terms are gated by the minutes prior. A separate start probability is not in the formula. Red cards, own goals, and missed penalties are not separate terms. Defensive contributions are added for every season in the cache, including seasons whose official points did not have that award.

## What the review got right

The three-week value reuses this week's score. It does not rebuild the rate for the next opponent. A future week in which the club has no fixture is already set to 0 for that week only. A blank this week is different: today's score is set to 0, and that 0 is what the later weeks copy, even when the club plays. Doubles are scored once. `fixture_counts` is not passed in.

The opening 15 maximises the sum of all fifteen scores. Every later week maximises the XI only, and the bench is worth 0 in that value. Those are different objectives. Replacing the opener with an XI-only sum, and leaving the bench worth nothing, was rejected. Under a £100m cap that buy fills the bench with the cheapest bodies, and the season then spends free transfers repairing it. Autosubs are the reason that matters: in 2023/24 they covered 65 of 86 intended blanks, and in 2024/25 they covered 38 of 42.

The search only tries same-position swaps, then a second swap from the best six, then a third only if two is ahead. A cross-position pair exists in the code and is switched off.

5-2-3 is a legal shape and is absent from the published list. Putting it into that list would move every published total. A count, with the list left as it is, does not.

The 45-minute buy rule is a gate, not a minutes forecast. Historical injury, doubt, and suspension are not on these sheets in a form known before the deadline. A live season can store them only as a snapshot taken before that deadline.

## What the second review added

Mean error can be near zero while over- and under-prediction cancel. The calibration table therefore also records MAE, RMSE, Pearson correlation, Spearman rank correlation, and the mean actual points in each decile of the predicted score. The leave-alone rule is unchanged: mean error inside 0.10, and removal moves Spearman against total points by less than 0.01. The extra columns are the reading, not a second kill rule. Rank correlation is the column that matches how the search buys.

The later bench objective, when it is designed, is the chance the starter blanks, times the chance the bench player plays, times the points expected if he plays. That is not a crude minutes floor, and it is not in this batch. The opener stays the sum of fifteen.

A count of 5-2-3 records how often it wins, the mean score advantage when it wins, and the actual points. The published list stays.

The cross-position check records, each week, the best exhaustive squad value minus the value the current search kept. A mean gap of about 0.1 to 0.2 per week is not worth a search rewrite.

The penalty of 1.0 is not treated as proved optimal. It stays frozen. A later grid, if one is run, walks forward across seasons. It does not pick the number that wins a single season.

Player goal, assist, and goal-or-assist prices are not on disk. No call is made to fetch them. A later comparison, once a snapshot exists, is team scoreline against that scoreline plus the player markets, and then plus the historical share. Those prices would be de-vigged, and the goal-or-assist price would be kept as the joint rather than built by assuming the two events are independent. Until the file exists, that experiment stays parked.

A double is the sum of the two fixture projections, with a chance of playing each. It is not a doubling of one score, and it is not part of the horizon change.

## What stays as it is

The hold margin of 1.25 and the penalty of 1.0 are not retuned. Penalty 0 was ahead on one season and behind on the next. Penalty 2.0 lost all three earlier seasons and left more blank starters. The discount of 0.9 stays. Chips stay off in the historical climb. The empty week in 2022/23 stays skipped, and the free-transfer count does not move, because that week was not played under its own number. The three-player club cap is already enforced on every candidate squad.

## Order

**0. Split the live gap, no formula change.** Done for Gameweeks 1–5 from ojaminFC's opening fifteen. With the price still used as a score, the path was 313 against 350. Of that −37, the captain extra was −21, the final elevens were −14, and their Triple Captain premium was −2. Hits were zero. M.Sangaré was captain for the first three weeks because a missing `score_xp` was filled with his price, and the search value sat near 680.

Feature rows that already pass the history gate have no missing `score_xp` in 2022/23, 2023/24, 2024/25, or 2025/26. The price was used for owned players who never enter that table. Sangaré's first feature row is Gameweek 4.

The fill is now 0. The price stays the budget column. Gemini kept that repair. The same five weeks, scored again, are 336 against 350. The gap is −14: the final elevens are +1, the captain extra is −5, their Triple Captain is −2, and the model's hit charge is −8. Gameweek 2 sells three players, including Sangaré, and pays for two hits. Gameweek 5 still holds while ojaminFC brings in Calvert-Lewin and Barnes. Five weeks remain a diagnosis. See `reports/stage_40_gw15_gap.md`.

**0A. Calibration table, no formula change.** Done for 2025/26, Gameweeks 5–38, on the buy pool (7,569 rows). The score is high by 0.13 points per row. MAE is 2.29 and RMSE is 3.00. Pearson is 0.24 and Spearman is 0.23. Deciles rise together: the lowest tenth predicts 2.17 and scores 2.26; the top tenth predicts 5.11 and scores 4.80. The score ranks. It does not pin a single week's points. See `reports/stage_39_calibration.md`.

Clean sheets are high by 0.23 points, the largest component bias, and removing them moves Spearman by 0.023. Goals are almost unbiased in the mean (−0.02) and are the largest rank term (Spearman drop 0.046), with MAE 0.75 because returns are rare. Defensive contributions on this season are unbiased (−0.03) and help rank (drop 0.024), so they stay. The bonus proxy and the card proxy meet the leave-alone rule. Appearance is high by 0.16. A goalkeeper goal is still worth 10 inside `xp_goals` and 6 in the official total. That scale stays so earlier totals remain reproducible.

Defensive contributions before 2025/26 are not subtracted from this table. That season is the one in which the award exists. A phantom-points table is a separate count.

**0B. One change to the three-week value, after the price fill is gone.** Later weeks keep the decision-week share and minutes prior, and take that fixture's own pre-deadline scoring rate and clean-sheet probability from odds already on disk. Saves, goals conceded, and the bonus proxy move with those two inputs, because they are functions of them. A new defensive-contribution model against possession is not part of this change. A week with no fixture is 0 for that week only. A blank this week must not zero a later week that has a fixture. No double multiplier. No new odds call. γ, the hold margin, the penalty, and `score_xp` stay put. The screen is the 2023/24 free-transfer climb, Gameweeks 5–38, because that season contains the Gameweek 29 blank. The fast XI cannot see this change.

Park 0B if the season total is not higher, or if the total on weeks where all twenty clubs play is not higher. A gain that exists only on the blank weeks is not a pass.

2023/24 Gameweeks 5–38: the published path scores 1665 and the opening horizon scores 1734 (+69). On the 29 weeks where all twenty clubs play the gap is +84. The other weeks are −15. Opening prices cover every club-week that has a sheet row. Gemini kept the pass. It does not replace the published value. The other seasons are the next gate, under the same rule, and a season whose opening-price coverage is below 0.98 is not run. See `reports/stage_41_open_horizon.md`.

Reading a future feature row is not allowed. That row's minutes prior includes matches after the deadline.

**1. Defensive contributions on earlier seasons, only after their own table.** Set that term to 0 for seasons before 2025/26 only if that table shows phantom points. No new parameter. The 2025/26 calibration does not justify it.

**2. The opener is not switched to an XI-only sum in this batch.** It stays the sum of fifteen. The later bench rule, if one is written, is the expected value of the automatic substitute. That rule is not designed here.

**3. Counts, not replacements.** How often 5-2-3 would be the best shape, the mean score advantage when it is, and the actual points, without changing the published list. One season with the cross-position two-transfer switch on, against the current beam, plus the per-week gap between the best exhaustive value and the value the beam kept. Availability and the minutes gate wait for a snapshot taken before each deadline.

**4. Player goal and assist prices, only when a snapshot is already on disk.** No fetch. De-vigging and the joint goal-or-assist price belong to that experiment.

**5. The penalty and the hold margin, only after the horizon has a result.** A walk-forward grid. Not a one-season maximum.

**6. Automatic substitutes, doubles, and chips.** A double is the sum of two fixture projections. Chips stay off in the historical climb until the forecast path is the thing being measured.

## Not in this batch

A new attacking model. A refit of the hold margin or the transfer penalty before 0B. A chip search. An autosub term inside the squad value. An injury flag built from minutes or from the end-of-season status file. A double-gameweek multiplier in the same run as the horizon change.
