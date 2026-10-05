# Architectural state

Updated: 2026-10-05. Stage history is root `CONTEXT.md`. Parked hypotheses are `.cursor/ideas.md`. This file is the plan, the work already done, and the direction from here.

## Plan

The engine is a pre-deadline squad advisor under 2026/27 rules. The published comparison stays the free-transfer climb on `score_xp` with an empty chip map, so older totals stay side by side. A new player score is screened on the fast XI. A change to the transfer rule is tested on the free-transfer climb. Chip choice sits above the points model: the caller prices the weeks, and `plan_half` leaves `score_xp` as it is.

The live path advises the stored ojaminFC squad at the next deadline. It uses that squad's bank, free transfers, and chips already played. A chip total from that path stays out of the published comparison. An Odds API call needs a cost estimate and the PI's approval. One trial has already been spent. See `.cursor/rules/odds-api-quota.mdc`.

Cursor writes and runs the code. Gemini locks a formula before it is coded and reviews a result before that result is called reviewed. The PI reads a batch.

## Fixed

- `score_xp` is the published score. Hold margin 1.25. Switch penalty 1.0. Buy gate: three prior appearances and expected minutes at least 45. A missing score is 0.
- The three-week value is the opening-price horizon. The current week keeps `score_xp`. A later week uses that fixture's opening 1X2 and the 2.5 total. A double stays one fixture. An owned player with one or two prior appearances keeps a past-only score capped at 6 and cannot be bought.
- The published climb plays a chip only when a week map is passed. The empty map is the published path. `plan_half` is the half-season planner: Bench Boost on the best week still in the half, Free Hit margin 12 on that week alone, Wildcard margin 16 on the rebuilt eleven against the held eleven across the rest of the half. Those margins stay. `src/live/policy.py` is the earlier single-week rule and is called from tests. The deadline scorer calls `plan_half`.
- A goalkeeper goal inside `compute_xp` is 10 points. The official award is 6. The published formation list omits 5-2-3. Both stay so older totals remain comparable.
- Solver constants live in `src/rules/fpl_2026.py`. Historical targets are Vaastav `total_points` for that season, not a 2026/27 rescore.

## Done

### Score and search

Variance, ownership, a heavier switch penalty, scheduled minutes, seven score repairs, and crowd flow were screened and parked. `score_xp` remains the published score. The 2025/26 calibration on 7,569 buy-pool rows is bias +0.13, MAE 2.29, Spearman 0.23. Clean sheets are the large positive bias. Defensive contributions on this season stay. See `reports/stage_39_calibration.md`, `reports/stage_45_score_repair.md`, `reports/stage_36_sched_stability.md`, `reports/stage_37_penalty2.md`, `reports/stage_47_crowd_context.md`, and `reports/stage_48_crowd_calls.md`.

### Gameweeks 1–5

ojaminFC scored 350. From their opening fifteen:

- The 327 in `reports/live_benchmark_2026.md` carried a later week's score into an earlier XI. The figures to use are the ones below.
- Past weeks only, while a missing score was still filled with the price: 313.
- The same freeze after that fill became 0: 336 (`reports/stage_40_gw15_gap.md`).
- The published path, opening horizon plus the early score: 333, with no hit (`reports/stage_42_gw15_horizon.md`). The horizon without the early score is 366.

A fresh squad on the same rule scores 263 against 350. Fourteen managers were locked before their squads were opened. The model's sales and the managers' own sales are both bought ahead. That is the variance reading. See `reports/stage_46_transfer_gap.md`.

### Crowd squads and chips

Four legal Gameweek 1 fifteens per season come from that week's ownership: template, premium, next, and third. Points are not used to pick them. Empty-chip climbs: the best is the template in 2022/23 (1735), 2023/24 (2105), and 2024/25 (2086), and Next in 2025/26 (2171). The 2025/26 template hold is void. Sale margins on the four leading climbs are inconclusive, and the hold stays 1.25. See `reports/crowd_openings.md`, `reports/crowd_opening_scores.md`, and `reports/crowd_sale_margins.md`.

`plan_half` on those fifteens is a side report. 2024/25 is absent. Every squad wildcards in Gameweek 4. 2022/23 loses on every squad, best lift Third at −40. 2023/24 best lift is Third at +128. 2025/26 gains on every squad. Quote Next +34 (2171 to 2205). The highest chip total is Premium 2214. Template's lift is +199. Free Hit is played once, on the 2023/24 Third squad in Gameweek 30. Gemini kept the count. The empty-chip file is unchanged. 2022/23 and 2023/24 use the 2026 wallet. 2025/26 is the matched wallet. See `reports/half_plan_scores.md`.

### Live deadline, Gameweek 6

The deadline is 10 Oct 2026. Entry 2632584, ojaminFC, has 350 points through Gameweek 5, £1.5m in the bank, 1 free transfer, and Triple Captain already played in Gameweek 1. Purchase price is the Gameweek 1 value, or `in_cost` for a later buy. The stored picks have no site selling price, so the formula is used. Calafiori was bought at 55, is now 58, and sells at 56.

`data/live/gw_lines.csv` holds Gameweeks 6 and 7 from one Odds API request: region `us`, markets `h2h` and `totals`, 2 credits, 498 remaining. Ten matches in each gameweek. Nine US books in Gameweek 6 and five in Gameweek 7. Chelsea–Bournemouth, Manchester United–Tottenham, Brentford–Liverpool, and Manchester City–Ipswich have no 2.5 total, so those cells are blank and the pot stays even. A 3.5 or 2.75 price stays out of the 2.5 column. The historical football-data file is unchanged.

`src/live/scorer.py` wraps `xp_on_pot`, the same one-match formula as `compute_xp`. Minutes come from a `player_id, gw, xmi` file. A written 0 stays 0. A player the file omits keeps his last observed minutes and is labelled `no_news`. Shot shares stay at the deadline. Each priced week uses that week's opening pot, and a double uses the first pot. Gameweek 8 has no 1X2, so a run repeats Gameweek 7 and later club weeks copy that step. One rebuild is paid from the bank plus sales. `plan_half` then reads the table. The transfer search is left for a later pass.

No minutes file was passed. The scorer was not run. No chip was chosen. The 12 and 16 hurdles were not applied. The note is `reports/live_deadline_gw6.md`. This is not a reviewed chip recommendation.

## How the direction moved

The outline asked for objectives beyond raw expected points: variance, ownership, transfer stability, and bench weight. Those screens lost, changed sign, or missed their bar. `score_xp` stayed the published score.

The Gameweek 1–5 gap against ojaminFC was the next diagnosis. A missing score filled with the price, and a stub copied a future week into an earlier XI. After those were closed, the opening horizon and a capped early score became the published three-week value. The transfer count against the locked managers showed both sides bought ahead.

Chips were the following lever. The half-season planner is built, and the crowd side report sits beside the empty-chip climb. The margins were left where they were. The published climb still passes no chip week.

The build is now the live week. The historical formula is the inner call. Around it are the squad already owned, a minutes sheet, this week's line, and one chip decision for the half.

## Next

`src/live/minutes_llm.py` is the minutes model. One completion covers the owned fifteen and every player who is doubtful, injured, suspended, unavailable, or given a chance below 100. Last observed minutes are context. A question, a missing id, minutes outside 0 to 90, or minutes above 0 for someone who cannot play writes no file. The scorer is not called from that module.

One call was made on the 2 Oct 2026 bootstrap. All 227 required rows passed. The sheet is `data/live/xmi_gw6.csv`, which stays out of git. Players with no flag and no place in the fifteen are absent from it. See `reports/live_minutes_gw6.md`.

The scorer has now read that sheet. The locked sum plays Wildcard in Gameweek 6. The rebuilt eleven leads by 12.56 this week and by 5.11 in Gameweek 7, and Gameweeks 8–19 repeat Gameweek 7, so the half-season sum is about 79 against the hurdle of 16. Gemini reviewed that table ([wildcard arithmetic](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)) and kept it as the arithmetic only. It is not a reviewed recommendation. The later Free Hit and Bench Boost in the schedule are the rest of today's winning combo. They are not played now. See `reports/live_deadline_gw6.md`.

Still parked: an Asian handicap as a live-week pot only, player props after a fresh cost estimate and the PI's approval, a shrunk season-rank weight on weeks that have no 1X2, and the other two guider outputs (a search constraint, or a question back). Those outputs leave `score_xp` unchanged. The 2025/26 top-100 archive remains a descriptive benchmark, not a sample to fit. Assistant Manager, and the 2024/25 chip wallet, stay out until that chip exists.
