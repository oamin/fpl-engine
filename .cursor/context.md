# Architectural state

Updated: 2026-10-04. Stage history lives in root `CONTEXT.md`. This file is the living state for the execution engine.

## What exists

Predictive stack is stages 0–27 (`src/models/`, `reports/`). Primary evaluation gate remains `src.models.season_climb`: each gameweek, a position-legal XI by score, captain = top score, cumulative actual points versus baselines. Budgeted and free-transfer climbs (stages 18–19, 27) already enforce 2/5/5/3, £100.0m, ≤3 per club, and a free-transfer bank of 5 at −4 per extra transfer.

Those constraints now come from `src/rules/fpl_2026.py`. `season_climb_budget` and `season_climb_ft` re-export them. Chip logic, 2026/27 BPS deltas, and official match awards live in that module and are unit-tested. The free-transfer climb applies a caller-supplied chip map and does not choose the week. An empty map leaves the climb unchanged.

## 2026/27 rules encoded

- Squad, budget, club cap, sell price, FT bank, hit cost.
- Eight chips: Wildcard, Free Hit, Bench Boost, Triple Captain, one copy in GW1–19 and one in GW20–38. Unused first-half chips expire at the GW19 deadline. One chip per gameweek. Wildcard and Free Hit do not spend the FT bank (wildcard does not reset it).
- Match points: appearance, goals (GK/DEF 6), assists, clean sheets, saves, cards, own goals, penalties, goals conceded, DefCon (DEF 10 CBIT, MID/FWD 12 CBIRT, 2 pts).
- BPS changes only: 1 BPS per 3 CBI, tackled deduction removed, GK saves +2 any / +1 inside the box / +1 big chance, penalty-save line 7 plus the big-chance point (net 8). Bonus 3/2/1 tie breaks.

## Known drift (not changed this step)

- `src/models/xp_engine.py` scores a goalkeeper goal as 10. Official award is 6. Left in place so stage-13+ xP is reproducible.
- `src/models/season_climb.py` `FORMATIONS` omits the legal 5-2-3. `OFFICIAL_FORMATIONS` in the rules module includes it. The climb list was not replaced, so published gates stay comparable.
- FT climbs play no chips unless a week map is passed. DefCon in the xP prior is gated on 60 minutes; the official award is not.
- Historical climb targets are Vaastav `total_points` under that season's rules, not a 2026/27 rescore.

## Collaboration

Cursor builds and runs. Gemini 3.8 Flash is the Co-PI: method design and diagnostic review, resumed across the thread for that milestone. The PI gets a mini-report only at a major milestone. Choice of the next method sits with Cursor and Gemini, inside `PROJECT_OUTLINE.md`.

Stage 30 left arm 2 inconclusive and killed the ownership player score. The search protocol is `src/models/search_protocol.py` with candidates in `experiments/matrix.json`. The equal blend (alpha=0.5) was +22 on the 2025/26 transfer climb, +187 on 2023/24, and −23 on the required 2024/25 holdout (1866 vs 1889). It is not carried forward. See `reports/search_protocol.md`.

Stage 31 tested switch penalties 0, 2, and 3 against the default 1.0. Zero penalty was +90 on 2025/26 and −55 on 2023/24. Penalty 2 was +46 on that screen. On 2022/23, 2023/24, and 2024/25 it was −128, −79, and −156, with more blank starters after substitutes in each. The default penalty stays 1.0. See `reports/stage_37_penalty2.md`.

Stage 32 put ownership weight 0.5 only inside transfer value. It scored 1840 against 1868 (−28) with the same transfer rate and more blank starters. No 2024/25 check. See `reports/stage_32_own_value.md`.

Stage 34 climbed agree_min, starter, minutes, and upside. Minutes was +118 on 2025/26 and −116 on 2024/25 (1773 vs 1889), with hits 14 against 9. Starter also cleared the screen (+63) and was not the check arm. No winner. See `reports/stage_34_follow.md`.

## Penalty shots

Vaastav has `penalties_missed` and `penalties_saved`, not penalties scored. `src/ingest/understat_penalties.py` adds `penalties_taken`, `penalties_scored`, and `penalty_xg` to the cached gameweek sheets for 2022/23–2025/26. The source is Understat's post-match shot feed (`situation == Penalty`). A shot is written only when the matchday club has exactly one matching player. Misses on those taker rows match Vaastav `penalties_missed` in each of the four seasons. The columns are not a pre-deadline designated-taker list, and `score_xp` is unchanged. See `reports/penalty_sheets.md`.

## Live planner

`src/live/` collects the free FPL bootstrap and fixtures for the next deadline. Expected minutes arrive as a `player_id, gw, xmi` file; a missing file picks no team and does not use the historical rolling minutes. Odds are a snapshot already on disk. The Odds API is not called. Captain and bench follow the supplied score. The XI list is the official one, including 5-2-3, and the historical climb list is unchanged. The chip rule is in `src/live/policy.py`. As of the GW6 deadline on 10 Oct 2026 the slate is a single gameweek, so that rule plays no chip until a minutes file and a score exist.

## Scheduled minutes

`score_xp_sched` keeps bench weeks in the minutes prior and leaves the shot share alone. It does not replace `score_xp`. Across GW5–38 the free-transfer gap is +9, −78, +146, and −44 for 2025/26, 2022/23, 2023/24, and 2024/25. The +34 replace-the-score bar was left where it is. It was not what hid a stable edge: the transfer gap changes sign. The fast XI stays within 4 points on 2025/26 and is ahead on the earlier seasons. See `reports/stage_36_sched_stability.md`.

## Blank weeks

A club with no row in the week is ranked at 0, and a week with no clubs is skipped. The hold and the chosen move record the XI sum and the no-fixture count for each week of the three-week window. On 2023/24 this scored 1665 against the previous 1676, with Gameweek 29 blanks falling from 8 to 2. `score_xp` and the transfer penalty are unchanged. See `reports/stage_38_blank_context.md`.

## Forward plan

An outside review of the gameweek note is answered in `reports/forward_plan.md`. `score_xp` already includes clean sheets, saves, defensive contributions, a bonus proxy, and deductions. The published three-week value no longer reuses today's score, and a blank this week does not copy that 0 onto a later week that has a fixture.

The Gameweek 1–5 gap from the same opening fifteen was 313 against 350 while a missing `score_xp` was filled with the price. That fill is now 0. The same five weeks are 336 against 350 (`reports/stage_40_gw15_gap.md`). The final elevens are one point ahead, the captain extra is −5, their Triple Captain is −2, and Gameweek 2 pays an eight-point hit. The published three-week value is the opening-price horizon. The current week keeps `score_xp`. A later week uses that fixture's opening line. A double stays one fixture. The holdout parked the sum of the two pots. An owned player with one or two prior appearances keeps a past-only score, capped at 6, and he cannot be bought. On Gameweeks 1–5 from the same opening fifteen the freeze scores 336, the horizon without that score scores 366, and the published path scores 333 against ojaminFC's 350. See `reports/stage_42_gw15_horizon.md`.

Both fixtures of a double scored 1799 against 1734 on 2023/24, then missed the same rule on the holdout: 2022/23 −78, 2024/25 +33 with the other weeks −4, 2025/26 −65. Gemini kept the park. The published calendar stays one fixture. 5-2-3 is the best shape in none of 135 weeks. The cross-position pairs the beam skips are worth 0.06 a week on 2025/26. See `reports/stage_43_dgw_follow.md` and `reports/stage_44_search_counts.md`.

The 2025/26 calibration is done (`reports/stage_39_calibration.md`). On 7,569 buy-pool rows the score is high by 0.13, MAE is 2.29, and Spearman is 0.23. Clean sheets are the large positive bias. Defensive contributions on this season stay. The hold margin and the transfer penalty stay.

Seven repairs of that score were screened on the fast XI and parked (`reports/stage_45_score_repair.md`). Mean defensive-contribution points are 0 before 2025/26, because the sheet has no such column. Removing the real 2025/26 term costs 85 points. `appear_linear` is the best of that batch and is 29 behind in 2022/23. No arm was ahead on every season, so none took a free-transfer climb. `score_xp` stays the published score.

## Transfer gap, Gameweeks 1–5

Fourteen managers were locked before their squads were opened: seven with at least two top-10,000 finishes in 2022/23–2025/26, and seven current rank slots at rank_sort 100, 500, 1000, 5000, 10000, 25000, and 50000. ojaminFC is a reference row. Each published climb starts from that Gameweek 1 fifteen. Gross is the sold player's points over the sale week and the next two, minus the bought player's points. Positive would mean the sale was the wrong way. A sign is consistent when at least four managers share it and that side is at least twice the other. The systematic reading would have been the model's sales sharing a sign while the managers' own sales did not.

Veterans: 17 model sales, mean gross −4.94, and the nine the manager still held averaged −2.89. Rank slots: 17 sales, mean −2.53, kept subset −2.83. Both groups, and the managers' own sales, are bought ahead. That is the variance reading. The same names recur (Isak to Thiago from three openings, Tarkowski to Guéhi from three). Those repeated sales sit inside the negative mean. Wildcard and free-hit weeks are most of the managers' own sales; the ordinary weeks are bought ahead by more, not by the other sign. The reference row is the other sign: Cherki to Szoboszlai +8 and Ødegaard to Enzo +3, mean +5.50, and the climb is 333 against 350. The cohort climbs score about 321–353 against official totals of 367–429, chips included, with no hit. Gemini kept the count. `attack_minutes` stays parked. `score_xp` stays the published score. See `reports/stage_46_transfer_gap.md`.

## Crowd flow

A residual on transfers in, `score_xp` plus the within-week gap between the percentile of `log1p(transfers_in)` and the percentile of `score_xp`, was dropped on 2026-10-04 before any code. It takes about half a point off a player everyone already owns, and it adds up to a point to last week's scorer.

The context check that replaced it is parked (`reports/stage_47_crowd_context.md`). Flow and ownership stayed in their own columns. On Gameweeks 5–38, leaving out one season at a time, they lowered the points error by 0.011 on average, short of the locked 0.02. The error fell in every season. The interaction raised it in every season. The in-sample slopes are not the score.

The close-call count is closed (`reports/stage_48_crowd_calls.md`). On players within 0.25 of the position leader, the same slopes picked the higher scorer on 57%, 43%, 43%, and 50% of decisive flips. The point gap changed sign. `score_xp` stays the published score.

## Half-season chips

`plan_half` in `src/live/half_plan.py` searches the remaining half at each deadline, with `score_xp` as the points the caller already priced. Bench points enter on the planned Bench Boost week inside that sum. A chip left unused at Gameweek 19 is worth nothing. It does not replace the live chip rule. The transfer search scores that same bench on the planned week only, when the caller passes it: discounted inside the next three weeks, or once from the decision-week bench when the chip is further out. The published climb passes no week, so its bench stays at zero. After the chip is used the week is dropped and the ordinary hold sells the bench down. A double stays one fixture. See `reports/half_plan.md`.

## Crowd openings

No public archive holds hall-of-fame Gameweek 1 squads for 2022/23–2025/26. Four legal fifteens per season are drawn from that week's ownership: the template, the dearest names in that same pool, then two later waves that share nobody with the template. The share is of managers. Points are not used. See `reports/crowd_openings.md`.

Those fifteens were scored on the published rule with an empty chip map (`reports/crowd_opening_scores.md`). The best climb is the template in 2022/23 (1735), 2023/24 (2105), and 2024/25 (2086), and Next in 2025/26 (2171). In 2022/23 the template and the premium both finish 51 below their hold. The 2025/26 template hold failed: Marc Guiu's move to Chelsea left four Chelsea players, and Gameweek 4 spent three free transfers. That sum is not a baseline. The template climb that season is 1988. Gemini kept the count. `score_xp` stays the published score.

The margins on the four leading climbs are inconclusive (`reports/crowd_sale_margins.md`). The 2022/23 template second half, which gave back 66 points, had a median margin of 4.47 and 1 of 16 sales under 2.5. The 2023/24 second half, which added 197, had a median of 3.49 and 4 of 13 sales under 2.5. A tighter hold would cut the paying half harder. The hold margin stays 1.25.

## Lineup

Each week the XI is the legal shape with the highest sum of `score_xp`, taking the top score in each position. The published shape list omits 5-2-3 so older totals stay comparable. The bench is the other four, ordered by that same score, and an automatic substitute skips anyone who did not play or who would break the shape. The published transfer search scores the XI and scores the bench at zero. A caller who names a Bench Boost week scores the bench on that week only. The whole week, from the empty-sheet check through the free-transfer bank, is drawn in `data/plots/gw_decision_flow.png`. Six patches to that stream were parked on 2026-10-03. See `.cursor/ideas.md`.

## Availability

Injury and doubt are known before the deadline, and they are a valid reason to leave a player out. The 2022/23–2025/26 cache cannot carry that fact. `merged_gw` is the match. `players_raw` is one end-of-season status row: De Bruyne's 2023/24 absence is missing from it, and a May injury flag would be copied back onto weeks when the player was fit. A zero-minute week is not an injury label. The live bootstrap may be used for a future deadline only if that deadline has its own snapshot taken before the deadline. Nothing was added to the climb. See `.cursor/ideas.md`.

## Stub scores

A player with no row in the decision week used to inherit his latest score in the whole frame, including a later week. Davis and Sangaré were named in the Gameweek 1–3 XI on a Gameweek 4 score. The carry is now limited to earlier weeks. The 327 in `reports/live_benchmark_2026.md` is the run that had the leak. The same published rule, past weeks only, scores 313 on those five weeks, keeps João Pedro, and takes no hit.

## Opening-price horizon

`python -m src.live.open_horizon` is an older diagnostic. Its on-disk note scores 331 against the leaked 327. That number is not the published path. The published climb now prices later weeks from the opening line itself, with a double still counted as one fixture, and an owned player below the history gate keeps a past-only score capped at 6. Gameweeks 1–5 on that path are in `reports/stage_42_gw15_horizon.md`.

## 2026/27 benchmark

`python -m src.live.benchmark` runs the published xp free-transfer climb from Gameweek 1 through 5, with an empty chip map. Priors are 2025/26 rows linked by Opta code and stored before this season, then dropped. A later 2026/27 week is not used to fill a Gameweek 1 prior. The club on each row is the club they played for that week. A change of club is a transfer and counts toward the three-player cap. The model scored 263 and ojaminFC scored 350, residual −87. Haaland's Triple Captain added 2 points. The same rule started from their Gameweek 1 fifteen scores 327, residual −23. It is 5 ahead after Gameweek 2, then sells João Pedro and Cherki on a hit in Gameweek 3 and finishes 23 behind. Last season's shot share was not shrunk, and no book term was added on top of share times λ. See `reports/live_benchmark_2026.md`. The score was not retuned.

## Odds

Free-tier Odds API stays expensive. Prefer snapshots, FPL, and football-data.co.uk. See `.cursor/rules/odds-api-quota.mdc`.
