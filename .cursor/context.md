# Architectural state

Updated: 2026-10-06. Stage history is root `CONTEXT.md`. Parked hypotheses are `.cursor/ideas.md`. This file is the plan, the work already done, and the direction from here.

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

The same twelve climbs were replayed on 2026-10-05 and matched that file. Ten of the twelve Gameweek 4 wildcards still clear 16 on the three priced steps. The chip week is positive on every climb, and in 2022/23 the weeks with no chip are the loss. All eight second-half wildcards in 2023/24 and 2025/26 miss 16 on the priced steps. Gemini kept the split ([chip audit](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). The margins stay 12 and 16. See `reports/chip_audit.md`.

### Live deadline, Gameweek 6

The deadline is 10 Oct 2026. Entry 2632584, ojaminFC, has 350 points through Gameweek 5, £1.5m in the bank, 1 free transfer, and Triple Captain already played in Gameweek 1. Purchase price is the Gameweek 1 value, or `in_cost` for a later buy. The stored picks have no site selling price, so the formula is used. Calafiori was bought at 55, is now 58, and sells at 56.

`data/live/gw_lines.csv` holds Gameweeks 6 and 7 from one Odds API request: region `us`, markets `h2h` and `totals`, 2 credits, 498 remaining. Ten matches in each gameweek. Nine US books in Gameweek 6 and five in Gameweek 7. Chelsea–Bournemouth, Manchester United–Tottenham, Brentford–Liverpool, and Manchester City–Ipswich have no 2.5 total, so those cells are blank and the pot stays even. A 3.5 or 2.75 price stays out of the 2.5 column. The historical football-data file is unchanged.

`src/live/scorer.py` wraps `xp_on_pot`, the same one-match formula as `compute_xp`. Minutes come from a `player_id, gw, xmi` file. A written 0 stays 0. A player the file omits keeps his last observed minutes and is labelled `no_news`. Shot shares stay at the deadline. Each priced week uses that week's opening pot, and a double uses the first pot. Gameweek 8 has no 1X2, so a run repeats Gameweek 7 and later club weeks copy that step. One rebuild is paid from the bank plus sales. `plan_half` then reads the table. The transfer search is left for a later pass.

The minutes sheet from the 2 Oct completion was read. The locked sum plays Wildcard in Gameweek 6. The rebuilt eleven leads by 12.56 this week and by 5.11 in Gameweek 7, and Gameweeks 8–19 repeat Gameweek 7, so the half-season sum is about 79 against the hurdle of 16. Gemini kept that as the arithmetic only. The note is `reports/live_deadline_gw6.md`.

## How the direction moved

The outline asked for objectives beyond raw expected points: variance, ownership, transfer stability, and bench weight. Those screens lost, changed sign, or missed their bar. `score_xp` stayed the published score.

The Gameweek 1–5 gap against ojaminFC was the next diagnosis. A missing score filled with the price, and a stub copied a future week into an earlier XI. After those were closed, the opening horizon and a capped early score became the published three-week value. The transfer count against the locked managers showed both sides bought ahead.

Chips were the following lever. The half-season planner is built, and the crowd side report sits beside the empty-chip climb. The margins were left where they were. The published climb still passes no chip week.

The build is now the live week. The historical formula is the inner call. Around it are the squad already owned, a minutes sheet, this week's line, and one chip decision for the half.

## Next

`src/live/minutes_llm.py` is the minutes model. One completion covers the owned fifteen and every player who is doubtful, injured, suspended, unavailable, or given a chance below 100. Last observed minutes are context. A question, a missing id, minutes outside 0 to 90, or minutes above 0 for someone who cannot play writes no file. The scorer is not called from that module.

One call was made on the 2 Oct 2026 bootstrap. All 227 required rows passed. The sheet is `data/live/xmi_gw6.csv`, which stays out of git. Players with no flag and no place in the fifteen are absent from it. See `reports/live_minutes_gw6.md`.

The scorer has read the minutes sheet. The locked sum plays Wildcard in Gameweek 6. Gameweeks 6 and 7 are about 17.7 against the hurdle of 16, and the printed half-season sum of about 79 repeats Gameweek 7. Gemini kept that as the arithmetic only ([wildcard arithmetic](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). The later Free Hit and Bench Boost in the schedule are the rest of today's winning combo. They are not played now. See `reports/live_deadline_gw6.md`.

The chip audit is the check on that rule. The chip week scores points. The 2022/23 loss is the squad in the weeks after the wildcard. Bounding the wildcard sum to the priced steps is the open formula change, and it has not been locked. See `reports/chip_audit.md`.

The live what-if names the two Gameweek 6 squads. The wildcard rebuild leads the one free transfer by 10.55 in Gameweek 6 and by 2.75 in Gameweek 7. The 12.56 in the deadline note is the same rebuild against the squad with no transfer. Gemini kept the note as the forecast for those two weeks ([two paths](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). See `reports/live_whatif_gw6.md`.

Each of Gameweeks 1–5 was also solved from the fifteen ojaminFC held before that deadline, then discarded. The model scores 332 against his 350. The gap is −18: captaincy −5, transfers 0, lineup −5, hits −8. Two weeks are level and three are behind, so the model did not beat these five decisions. Gemini kept the split ([reset gap](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). Five weeks remain too few to call the rule reliable. See `reports/reset_gap_gw15.md`.

The same reset with a free chip wallet scores 347 against 350. The gap is −3. Triple Captain is played in Gameweek 3 and Bench Boost in Gameweek 4. Wildcard and Free Hit stay unused: Gameweek 1 cannot play them, and no later priced horizon clears 16 or 12. Two weeks are ahead, so the wallet did not beat these five decisions. It is 15 points ahead of the mirrored reset. Gemini kept the split ([free wallet](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). See `reports/reset_chips_gw15.md`.

The same wallet was then given entry 1078627's Gameweek 1 fifteen and told to keep the squad. That team is Charcot Donetsk. The model scores 376 against his 368. The gap is +8, with 3 of 5 weeks non-negative, so that is a gain on these five weeks from his starting fifteen. Bench Boost is in Gameweek 1, outlook 9.27, realised bench 18, and he played the same chip for the same 18, so the week is level. Triple Captain is in Gameweek 3 on Haaland, which he also played, and the +9 that week is the transfers. Gemini kept the split ([friend start](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). The Gameweek 4 bench of 2 on the reset is not in this sum, and it did not add a floor. See `reports/friend_start_gw15.md`.

That carry was then run on the 14 managers locked on 4 October. The model is ahead of none of them. The mean gap is −27.43. Veterans are −17.29, all seven behind or level. Rank slots are −37.57. The model plays Bench Boost and Triple Captain only. Gemini kept the split ([cohort carry](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). ojaminFC, left out of the mean, is 372 against 350 on this same carry. See `reports/cohort_carry_gw15.md`.

The lineup piece of that gap is −198. It was tagged from the intended eleven and from score_xp. Shape, an unmatched shared starter, is −196. Same-position pairs where the model had the higher score are −17. There is no pair where the model had the lower score. Gemini kept the tags ([lineup cause](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). See `reports/lineup_cause_gw15.md`.

That shape figure splits into formation −163 and displacement −33. Formation is an extra player at the position in the eleven named before the deadline. Displacement is a shared starter left over after that extra count. The model named more defenders on 24 of the 39 weeks that have one of these starters. Gemini kept the split ([shape split](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). See `reports/shape_split_gw15.md`.

The same fifteens were then tried with only the eleven changed. Five midfielders sit 0.25 below the named score and score +1.54 points. That is inconclusive, so the picker stays. Three forwards score −0.89 and the picker stays. Gemini kept the calls ([shape trial](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). See `reports/shape_trial_gw15.md`.

The transfer piece of −180 was then tagged. Chip signings are −329, same-position pairs where the human had the higher score are −81, and the model's other unshared starters are +230. The chip tag is past −90, so on this window the squad gap is the chip reset. Gemini kept the split ([squad gap](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). No rule changes. The frozen rules are in `reports/model_manifest.md`. See `reports/squad_gap_gw15.md`.

The close-call follow-up asked whether five midfielders score more when they sit within 1 of the named score. On the carried fifteens that band is +0.85 over 33 weeks, and +0.38 for the veterans, so the window reading is present. The six weeks above 1.50, too few to apply the control, are +9.83, and the weeks within 0.25 are −1.33. On the fast eleven, Gameweeks 5–38, every season has 18 to 22 weeks with more than three from one club, and the legal close calls are 4, 4, 4, and 6. None of the four seasons clear. The 18 legal weeks average −0.39. Gemini kept the retirement ([close calls](bc-7121b96b-db3a-552d-ade8-335c01d37eda)). The picker stays. See `reports/midfield_close.md`.

The chip-signing deficit of −329 was then traced on the same carry. The model could already buy 83 of the 85 players. The wildcard lead it turned down has a Gameweek 4–5 median of 3.67 and a Gameweek 2–3 median of 8.75. None of the late weeks reach 12, so the hurdle is not the lever, and the three-appearance rule is not either. The rebuild that was turned down held 19 of those players, −100 points. The other 64 were left out, and the lowest rebuilt player at that position outscored them by a median of 0.31. Gemini kept that as a score miss ([wildcard lead](bc-7121b96b-db3a-552d-ade8-335c01d37eda)). No constant moves. The lineup gap does not share these players. See `reports/chip_lead_gw15.md`.

The ranked-lower price tag of −39 was then read on the same carry. Fourteen of the seventeen pairs have no sale in the fifteen that pays for them, −25, and one sits within 1.25, −7. Nine of those fourteen are Haaland. Two pairs, Palmer, were already in the list of 35, and the horizon declined them. Gemini kept that as budget ([price reach](bc-7121b96b-db3a-552d-ade8-335c01d37eda)). The hold stays 1.25. A second sale would fund six pairs that sum to −7, and that does not open a replay. The cohort chain stops. See `reports/sale_reach_gw15.md`.

The next measurement was the close-pair shape, run on 2026-10-06. On the four seasons before this one, the eligible leader and the closest player within half a point were counted. The leader wins 48.6%, 56.1%, 45.8% and 56.6% of the decisive pairs. The lower blank rate and the higher haul rate, from earlier weeks only, are both short of 80 decisive pairs in every season, and both miss a 5 point margin over the leader in three seasons. Gemini kept them parked ([close-pair shape](bc-7121b96b-db3a-552d-ade8-335c01d37eda)). No later screen. `score_xp` stays. See `reports/close_pair_shape.md`.

Three outside hypotheses were audited on 2026-10-06. The upper-tail reading was left off the queue: it is the haul arm already counted. Gemini locked the other two ([hypothesis audit](bc-7121b96b-db3a-552d-ade8-335c01d37eda)).

Both were run the same day. The search shadow averages 0.005, 0.004, and 0.101 a week on 2022/23, 2023/24, and 2024/25, with 0, 0, and 1 weeks at or above 1. A rewrite is rejected. The clean fill clears both bars in 2023/24 only. The early gaps are +10.18, −9.97, and −0.82, and the Gameweek 5–8 gaps against the published eleven are +3.75, −10.25, and +9.75. Cold start is rejected. Gemini kept both calls ([queued tests](bc-7121b96b-db3a-552d-ade8-335c01d37eda)). The score and the search stay. See `reports/queued_tests.md`.

The Gameweek 1–5 gap does not include chips still held. The model played two chips on every squad, Bench Boost and Triple Captain. The 14 played 2.5 on average. The model has 0.5 more first-half chips left. Second-half wallets are full on both sides. Closing −27.43 with that half chip would need it to be worth 55 points, and no such price is on file. Gemini left the points unchanged ([chip stock](bc-7121b96b-db3a-552d-ade8-335c01d37eda)). See `reports/chip_stock_gw15.md`.

Three of those 14 already hold the same chips as the model. The gaps are 0, −26, and −46. The two with the same chip in the same week average −13, in the lineup column. A larger pool was reviewed and not opened ([chip match](bc-7121b96b-db3a-552d-ade8-335c01d37eda)). See `reports/chip_match_gw15.md`.

Three early-season trials were reviewed on 2026-10-06. The FDR-and-share score and the ownership tie-break are rejected. The buy from Gameweek 2, for one or two appearances capped at 6, scored 1606, 1960, 2019, and 2018 against the template climbs of 1735, 2105, 2086, and 1988. The pool is 7603 against 7948. Gemini kept the rejection ([early buy](bc-7121b96b-db3a-552d-ade8-335c01d37eda)). The buy gate stays at three appearances. See `reports/early_buy.md`.

The 23 model starters who played 0 minutes are not one fault. Fourteen are João Pedro in Gameweek 5, named on a carried 5.30 after four 90-minute weeks. The stored human slot is the eleven after automatic substitutes. Four of his seven owners had started him, including Jess Bernstein as captain, and a midfielder replaced him. Three had benched him. The submitted human elevens contain 10 zeros, not 0, and all 10 were replaced. A three-forward eleven replaces a zero-minute forward with the next midfielder or defender who played. Three are Rico Lewis in Gameweeks 3–5, still on his Gameweek 1 score, because a 0-minute row is dropped before the prior updates. The other six are the first week that player recorded 0. See `reports/zero_minutes_gw15.md`.

Still parked: an Asian handicap as a live-week pot only, player props after a fresh cost estimate and the PI's approval, a shrunk season-rank weight on weeks that have no 1X2, and the other two guider outputs (a search constraint, or a question back). Those outputs leave `score_xp` unchanged. The 2025/26 top-100 archive remains a descriptive benchmark, not a sample to fit. Assistant Manager, and the 2024/25 chip wallet, stay out until that chip exists.
