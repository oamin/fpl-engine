# Candidate hypotheses

Cursor and Gemini choose the next one. The PI is not asked to pick.

## Objective functions (from the project outline)

1. **Variance-aware utility.** `u = score_xp / (σ + 1)` is parked (stage 28, −253). `score_xp − 0.25σ` as the player score lost the fast screen (stage 29). The same penalty only inside transfer value scored +90 on 2025/26, then +79, +8, and −56 on 2023/24, 2024/25, and 2022/23. Exactly one prior season cleared +34, so the result is inconclusive and is not the gate. A swap penalty of 2.5 scored −13.
2. **Ownership-adjusted yield.** Deadline `selected` gives `ow = 15 * selected / sum(selected)`. As a player score it lost the fast XI by 124 (stage 30). Inside transfer value only, weight 0.5, it lost the 2025/26 free-transfer climb by 28 with more blank starters (stage 32). Not the transfer rule.
3. **Transfer-churn penalty.** Stage 31 locked penalties 0, 2, and 3 against the default 1.0. Zero penalty was +90 then −55 on the holdout, with more hits and more blank XI slots. Penalty 3 lost 81 on the screen. Penalty 2 cleared the screen by 46 and was not sent to the holdout, because only the best passer was. The default penalty of 1.0 stays. Do not hold out penalty 2 after that result.
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

## Rules gaps

- Full Opta BPS table (goals, assists, playing time, and the rest) is not in the 20 Jul 2026 change note. Only the 2026/27 deltas are coded.
- Chip mechanics are wired. `run_ft_season(..., chips={gw: name})` plays that week. An empty map plays nothing and does not choose a week. Wildcard and Free Hit spend the bank plus sell prices, not a fresh £100.0m. A kept player's purchase price stays. Free Hit reverts the squad. Bench Boost adds players left out of the final XI after automatic substitutions. Triple Captain is ×3, and a blank captain passes that ×3 to the vice-captain. The historical climb does not search chip weeks.
- Live chip policy (2026-10-02, Gemini): expected points only, current half, candidates are the current week plus blanks and doubles. Free Hit needs a 12-point edge on a blank. Wildcard needs a 16-point edge. Bench Boost and Triple Captain fire only on the best double, and only if that double is this week. A tie plays nothing. GW1 wildcard and back-to-back free hits stay illegal. Bench Boost in GW1 is legal. This does not rerun `xp_ft`.
- Rescoring historical seasons under 2026/27 BPS needs CBI, save location, and big-chance saves. Do not overwrite Vaastav `total_points` until those fields exist.
