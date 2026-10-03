# Architectural state

Updated: 2026-10-03. Stage history lives in root `CONTEXT.md`. This file is the living state for the execution engine.

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

## Lineup

Each week the XI is the legal shape with the highest sum of `score_xp`, taking the top score in each position. The published shape list omits 5-2-3 so older totals stay comparable. The bench is the other four, ordered by that same score, and an automatic substitute skips anyone who did not play or who would break the shape. The transfer search scores the XI and scores the bench at zero. The whole week, from the empty-sheet check through the free-transfer bank, is drawn in `data/plots/gw_decision_flow.png`. Six patches to that stream were parked on 2026-10-03. See `.cursor/ideas.md`.

## Availability

Injury and doubt are known before the deadline, and they are a valid reason to leave a player out. The 2022/23–2025/26 cache cannot carry that fact. `merged_gw` is the match. `players_raw` is one end-of-season status row: De Bruyne's 2023/24 absence is missing from it, and a May injury flag would be copied back onto weeks when the player was fit. A zero-minute week is not an injury label. The live bootstrap may be used for a future deadline only if that deadline has its own snapshot taken before the deadline. Nothing was added to the climb. See `.cursor/ideas.md`.

## Stub scores

A player with no row in the decision week used to inherit his latest score in the whole frame, including a later week. Davis and Sangaré were named in the Gameweek 1–3 XI on a Gameweek 4 score. The carry is now limited to earlier weeks. The 327 in `reports/live_benchmark_2026.md` is the run that had the leak. The same published rule, past weeks only, scores 313 on those five weeks, keeps João Pedro, and takes no hit.

## Opening-price horizon

`python -m src.live.open_horizon` is a diagnostic arm, not the published climb. The current week keeps `score_xp`. Later weeks in the three-week hold use that fixture's opening 1X2, with share and minutes frozen at the deadline. On the ojaminFC Gameweek 1 fifteen it scores 331 against the published 327 and their 350. It does not sell João Pedro in Gameweek 3. His step scores that week are 2.94, 5.36 and 3.82. Gemini reviewed the run. These five weeks do not accept or reject the arm. See `reports/open_horizon_gw1_5.md`.

## 2026/27 benchmark

`python -m src.live.benchmark` runs the published xp free-transfer climb from Gameweek 1 through 5, with an empty chip map. Priors are 2025/26 rows linked by Opta code and stored before this season, then dropped. A later 2026/27 week is not used to fill a Gameweek 1 prior. The club on each row is the club they played for that week. A change of club is a transfer and counts toward the three-player cap. The model scored 263 and ojaminFC scored 350, residual −87. Haaland's Triple Captain added 2 points. The same rule started from their Gameweek 1 fifteen scores 327, residual −23. It is 5 ahead after Gameweek 2, then sells João Pedro and Cherki on a hit in Gameweek 3 and finishes 23 behind. Last season's shot share was not shrunk, and no book term was added on top of share times λ. See `reports/live_benchmark_2026.md`. The score was not retuned.

## Odds

Free-tier Odds API stays expensive. Prefer snapshots, FPL, and football-data.co.uk. See `.cursor/rules/odds-api-quota.mdc`.
