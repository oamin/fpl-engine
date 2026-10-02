# Architectural state

Updated: 2026-09-30. Stage history lives in root `CONTEXT.md`. This file is the living state for the execution engine.

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

Stage 31 tested switch penalties 0, 2, and 3 against the default 1.0. Zero penalty was +90 on 2025/26 and −55 on 2023/24. Penalty 2 also cleared the screen and was not the holdout arm. The default penalty stays. See `reports/stage_31_churn.md`.

Stage 32 put ownership weight 0.5 only inside transfer value. It scored 1840 against 1868 (−28) with the same transfer rate and more blank starters. No 2024/25 check. See `reports/stage_32_own_value.md`.

Stage 34 climbed agree_min, starter, minutes, and upside. Minutes was +118 on 2025/26 and −116 on 2024/25 (1773 vs 1889), with hits 14 against 9. Starter also cleared the screen (+63) and was not the check arm. No winner. See `reports/stage_34_follow.md`.

## Penalty shots

Vaastav has `penalties_missed` and `penalties_saved`, not penalties scored. `src/ingest/understat_penalties.py` adds `penalties_taken`, `penalties_scored`, and `penalty_xg` to the cached gameweek sheets for 2022/23–2025/26. The source is Understat's post-match shot feed (`situation == Penalty`). A shot is written only when the matchday club has exactly one matching player. Misses on those taker rows match Vaastav `penalties_missed` in each of the four seasons. The columns are not a pre-deadline designated-taker list, and `score_xp` is unchanged. See `reports/penalty_sheets.md`.

## Odds

Free-tier Odds API stays expensive. Prefer snapshots, FPL, and football-data.co.uk. See `.cursor/rules/odds-api-quota.mdc`.
