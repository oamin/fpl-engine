# Candidate hypotheses

Cursor and Gemini choose the next one. The PI is not asked to pick.

## Objective functions (from the project outline)

1. **Variance-aware utility.** `u = score_xp / (σ + 1)` is parked (stage 28, −253). `score_xp − 0.25σ` as the player score lost the fast screen (stage 29). The same penalty only inside transfer value scored +90 on 2025/26, then +79, +8, and −56 on 2023/24, 2024/25, and 2022/23. Exactly one prior season cleared +34, so the result is inconclusive and is not the gate. A swap penalty of 2.5 scored −13.
2. **Ownership-adjusted yield.** Deadline `selected` gives `ow = 15 * selected / sum(selected)`. As a player score it lost the fast XI by 124 (stage 30). Inside transfer value only, weight 0.5, it lost the 2025/26 free-transfer climb by 28 with more blank starters (stage 32). Not the transfer rule.
3. **Transfer-churn penalty.** Stage 31 locked penalties 0, 2, and 3 against the default 1.0. Zero penalty was +90 then −55 on the holdout, with more hits and more blank XI slots. Penalty 3 lost 81 on the screen. Penalty 2 was +46 on 2025/26 and was then run on the three earlier seasons: −128, −79, and −156, with more blank starters after substitutes in each. The default penalty of 1.0 stays. See `reports/stage_37_penalty2.md`.
4. **Bench weight and formation.** Score the bench (and 5-2-3) instead of the XI only, aimed at Bench Boost weeks. Blocked on a chip-aware simulator.

Stage 34 is complete. agree_min −44 and upside −8 on the 2025/26 transfer climb. Minutes was +118 then −116 on 2024/25, with 14 hits against 9. Starter was +63 on the screen and was not checked, because only the best passer went to 2024/25. Do not check starter after that result. No winner.

## Penalty takers

The gameweek sheets now carry Understat penalty shots (`penalties_taken`, `penalties_scored`, `penalty_xg`). That is a post-match fact. It is not a designated-taker list from before the deadline, and it is not wired into `score_xp`.

Gemini reviewed a split on 2026-10-02 ([penalty split](bc-721d1cc3-4c05-586a-a1e8-c24f0aecfff3)) and parked it. No fast-XI screen. `xp_goals` stays `share_xG × λ`. Vaastav `expected_goals` already includes penalty xG, so a second penalty term counts the same kick twice. Subtracting Understat `penalty_xg` (~0.76) from Opta xG is unsafe: in 2022/23, 37 of 97 taker rows have Opta xG below that penalty xG. A team awards about 0.11–0.14 penalties per match, and an established taker's kicks are already inside the rolling share. The 35% goal residual is conversion variance. A new taker is still unknown until after the first kick.

## Scheduled minutes

Parked after four seasons, not only because 2025/26 was +9. Free-transfer gaps, GW5–38: 2025/26 +9, 2022/23 −78, 2023/24 +146, 2024/25 −44. Two seasons ahead, two behind, worst loss 78. The fast XI never lost by more than 4 and was ahead in the three earlier seasons, so the ranking among players who played is quiet and the squad path is not. About 15% of eligible rows move by more than 0.5. Do not replace `score_xp`. Do not put the column in `experiments/matrix.json`. The +34 bar stays the bar for a claim that a new score should replace the published one. See `reports/stage_36_sched_stability.md`.

## Opening-price horizon

Coded 2026-10-02 after Gemini locked the split ([horizon prices](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). `xp_open_odds_h3` keeps the published current-week score and prices later horizon weeks from opening 1X2. The Gameweek 1–5 diagnostic held João Pedro and scored 331 against the published 327. That window is not the test. The remaining test is a full historical free-transfer season. Do not retune γ, the hold margin, or the share on these five weeks.

## Last-season share versus the book

Parked 2026-10-02 after Gemini reviewed it ([share versus book](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). Do not shrink the 2025/26 shot share to close the Gameweek 1–5 gap against ojaminFC, and do not add a book term on top of `share × λ`. The team rate and the clean-sheet proxy already come from 1X2 and over/under. There is no player goal or clean-sheet price in that window, so the book cannot replace the share. A forced decay, only if later required, uses a fixed prior of 5 matches and replaces the share. It is not an extra term, and the weight is not fit on these five weeks. No fast XI screen: the player-prop score does not exist.

## Player status

Parked 2026-10-03 after Gemini reviewed the files ([availability leakage](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). An injury known before the deadline is a valid reason to leave a player out. The cached seasons cannot supply that fact.

`merged_gw` has minutes, cards, and `xP`. It has no status, chance of playing, or news. A zero-minute row in the same week is the match result, so it cannot be the label. A lagged zero still misses the first week of a new injury, because that player's previous week was an appearance, and it treats a rotation rest as an injury.

`players_raw_{season}.csv` is one bootstrap row per player, scraped at the end of the season. In 2023/24 De Bruyne has an 18-week club-played hole from GW2 to GW20 and the season-end row is available, with empty news. The other direction is the same file: 46 of the 74 players marked injured at the end of 2023/24 had already played at least 400 minutes in GW1–15, including Cash, Ederson, and Dunk. Copying that flag onto the autumn benches them for an injury that had not happened.

A red card is on the sheet, and the player is usually on 0 minutes in the next club week that has a row (22 of 28, 53 of 55, 40 of 49, 38 of 43). That tag stays parked. A domestic cup can absorb the ban, an appeal can clear it before the next deadline, and the sheet does not say whether the ban is one match or three. There are 30 to 58 reds in a season.

A later live season can store status only as an append-only bootstrap snapshot taken before that gameweek's deadline, keeping status, both chance fields, news, `news_added`, and the capture time. The live snapshot today overwrites one file. Do not backfill a past week from a later dump. Nothing was coded.

## XI and bench selection

Parked 2026-10-03 after Gemini reviewed the stream ([lineup stream](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). Nothing in the batch is worth a climb. `score_xp`, the transfer penalty, and the published formation list stay as they are.

The weekly search scores the XI only. `pick_xi` then takes, for each shape, the top score in each position and keeps the shape with the highest sum. For an additive score that is the best XI. The other four are the bench: the goalkeeper, then the outfield by the same score. An automatic substitute already skips a player who did not play and a player who would break the formation, and that skip does not use the player up. The captain and the vice-captain are the top two scores. The vice-captain is paid only when the captain records zero minutes.

- Bench order by who can cover the lowest `p_play` starter. Killed. The autosub walk already skips an illegal first substitute. Putting a cheap defender first hands a midfield blank to that defender.
- Swapping a starter out for a cover player when the score given up is under an epsilon. Killed. It spends points every week on a blank the appearance-only minutes prior does not see coming.
- A vice-captain minutes bar. Killed. It does not touch the weeks where the captain plays and someone else hauls, and on a real captain blank it can hand the double to a low score.
- Adding a fraction of the first substitute into the transfer value. Killed. The weight is unfitted, and it spends budget and free transfers on the bench.
- A cap of two starters per club, lifted on a double. Killed. It benches a third attacker from a strong side and leaves that fee on the bench.
- Adding 5-2-3 to the published formation list. Closed. The count on the eligible pool, Gameweeks 5–38, is 0 wins in 135 weeks. The list stays as it is. See `reports/stage_44_search_counts.md`.

## Forward plan

Locked 2026-10-03 with Gemini after an outside review of the gameweek note ([forward plan](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). The note understated `score_xp`. The formula already sums appearance, goals, assists, clean sheets, defensive contributions, saves, a bonus proxy, and two deductions. The review's horizon point stands. See `reports/forward_plan.md`.

The second review of that plan is folded in. Calibration now records MAE, RMSE, Pearson, Spearman, and predicted-score deciles. The leave-alone rule is still mean error inside 0.10 and a Spearman move under 0.01. The later bench objective is the chance a starter blanks, times the chance the bench player plays, times expected points if he plays. That is parked. It is not a minutes floor, and the opener stays the sum of fifteen.

The Gameweek 1–5 split was 313 against 350 while a missing score was filled with the price. After that fill became 0, the same path is 336 against 350. The remaining −14 is the final elevens +1, captain extra −5, their Triple Captain −2, and a −8 hit in Gameweek 2. Gemini kept the fill ([score fill](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). A past-only score for an owned player below the history gate, if one is added later, belongs in the feature table, not inside the fill. See `reports/stage_40_gw15_gap.md`.

0A is done on 2025/26, Gameweeks 5–38, 7,569 buy-pool rows. Bias +0.13, MAE 2.29, Spearman 0.23. Clean sheets are high by 0.23. Goals are the largest rank term. Defensive contributions on this season stay. The bonus proxy and the card proxy meet the leave-alone rule. See `reports/stage_39_calibration.md`. Subtracting defensive contributions before 2025/26 still needs its own table.

The opening horizon is the published three-week value, approved after four seasons. A double is still one fixture on that path. An owned player with one or two prior appearances keeps a past-only score capped at 6. Gemini kept that guard after Gameweeks 1–5 scored 333 with it and 366 without it ([early score](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). The zero-score sale is not rewarded by those five weeks.

Both fixtures scored 1799 against 1734 on 2023/24. The holdout misses the same rule: 2022/23 −78, 2024/25 +33 with the other weeks −4, 2025/26 −65. Gemini kept the park ([double fixtures](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). The published calendar stays one fixture. A later double formula needs its own rule, locked before the totals are read. 5-2-3 wins no week on the eligible pool across four seasons. The unsearched cross-position pairs are worth 0.06 a week on 2025/26. Both counts are closed.

5-2-3 is counted, not added to the published list, and the count is closed at zero wins. The cross-position pairs are closed at 0.06 a week. Player goal and assist prices are not on disk, so that experiment is parked and no odds call is made. The penalty of 1.0 and the hold margin of 1.25 stay frozen. A later grid, if one is run, walks forward. It does not crown a one-season winner.

Seven score repairs were locked and screened on the fast XI, Gameweeks 5–38, four seasons ([score repairs](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). All are parked. `appear_linear` is the best of that batch and is still behind in 2022/23 by 29, so it does not take a free-transfer climb. Removing clean sheets is killed. Halving them, halving the goals-conceded deduction, and the defender/forward shift of 0.25 all change sign across seasons. A goalkeeper goal at 6 instead of 10 moves the fast XI by about a point. See `reports/stage_45_score_repair.md`.

`attack_minutes` was dropped on 2026-10-04 before any code ([transfer gap](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). It would have added a full clean sheet and a full defensive-contribution expectation for a midfielder or forward when those two were smaller than his goals and assists, leaving appearance as it is. A clean sheet needs 60 minutes. Scaling those awards up because the shots are large has no basis in the scoring rules, and the piece is too small to close the Cherki gap, which is mostly appearance and the team rate. The buy pool would have given a full clean sheet to players with 45 to 60 expected minutes. Before 2025/26 defensive contributions are already zero, so the arm would only have raised midfielder clean sheets. It was not put on the fast XI. `appear_linear` remains parked for the same sale: the 60-minute gate stays shut, and at 31 minutes that repair lowers appearance. `score_xp` stays the published score.

The Gameweek 1–5 transfer count against the locked managers was kept ([transfer gap](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). Model and managers are both bought ahead, so the pre-registered systematic reading does not fire. See `reports/stage_46_transfer_gap.md`.

Residual transfers in were dropped on 2026-10-04 before any code ([crowd flow](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). The locked formula would have added, inside each gameweek and only on the eligible pool, the gap between the percentile of `log1p(transfers_in)` and the percentile of `score_xp`, with the weight fixed at 1.0. A player everyone already owns has only middling transfers in and a score near the top, so the residual takes about half a point off him. The positive side is last week's scorer: a low score and a flood of transfers in adds up to a point. Residualizing last week's points as well leaves noise. The players a press conference might reveal, those below 45 expected minutes or with fewer than three prior appearances, are outside the eleven's pool. Raw net transfers already lost 104 (stage 23). Ownership as a penalty already lost the fast eleven by 124, and the same weight inside transfer value lost the 2025/26 climb by 28. A price change is that same flow a day later. No screen, no column, and no grid on the weight. `score_xp` stays the published score.

## Rules gaps

- Full Opta BPS table (goals, assists, playing time, and the rest) is not in the 20 Jul 2026 change note. Only the 2026/27 deltas are coded.
- Chip mechanics are wired. `run_ft_season(..., chips={gw: name})` plays that week. An empty map plays nothing and does not choose a week. Wildcard and Free Hit spend the bank plus sell prices, not a fresh £100.0m. A kept player's purchase price stays. Free Hit reverts the squad. Bench Boost adds players left out of the final XI after automatic substitutions. Triple Captain is ×3, and a blank captain passes that ×3 to the vice-captain. The historical climb does not search chip weeks.
- Live chip policy (2026-10-02, Gemini): expected points only, current half, candidates are the current week plus blanks and doubles. Free Hit needs a 12-point edge on a blank. Wildcard needs a 16-point edge. Bench Boost and Triple Captain fire only on the best double, and only if that double is this week. A tie plays nothing. GW1 wildcard and back-to-back free hits stay illegal. Bench Boost in GW1 is legal. This does not rerun `xp_ft`.
- Rescoring historical seasons under 2026/27 BPS needs CBI, save location, and big-chance saves. Do not overwrite Vaastav `total_points` until those fields exist.
