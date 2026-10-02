# Candidate hypotheses

Cursor and Gemini choose the next one. The PI is not asked to pick.

## Objective functions (from the project outline)

1. **Variance-aware utility.** `u = score_xp / (σ + 1)` is parked (stage 28, −253). `score_xp − 0.25σ` as the player score lost the fast screen (stage 29). The same penalty only inside transfer value scored +90 on 2025/26, then +79, +8, and −56 on 2023/24, 2024/25, and 2022/23. Exactly one prior season cleared +34, so the result is inconclusive and is not the gate. A swap penalty of 2.5 scored −13.
2. **Ownership-adjusted yield.** Deadline `selected` gives `ow = 15 * selected / sum(selected)`. As a player score it lost the fast XI by 124 (stage 30). Inside transfer value only, weight 0.5, it lost the 2025/26 free-transfer climb by 28 with more blank starters (stage 32). Not the transfer rule.
3. **Transfer-churn penalty.** Stage 31 locked penalties 0, 2, and 3 against the default 1.0. Zero penalty was +90 then −55 on the holdout, with more hits and more blank XI slots. Penalty 3 lost 81 on the screen. Penalty 2 cleared the screen by 46 and was not sent to the holdout, because only the best passer was. The default penalty of 1.0 stays. Do not hold out penalty 2 after that result.
4. **Bench weight and formation.** Score the bench (and 5-2-3) instead of the XI only, aimed at Bench Boost weeks. Blocked on a chip-aware simulator. Stage 39 tested the bench half without chips and without 5-2-3: weight 0.25 inside transfer value only. Screen +58 (1926 vs 1868) sits in the 2025/26 opening-squad band 1839–1972, so it is not an edge. 2024/25 was −152 (1737 vs 1889), hits 14 against 9. Not carried forward. Do not retune 0.25 and do not hold out a second weight. Formation 5-2-3 stays out until a chip-aware simulator exists.

Stage 34 is complete. agree_min −44 and upside −8 on the 2025/26 transfer climb. Minutes was +118 then −116 on 2024/25, with 14 hits against 9. Starter was +63 on the screen and was not checked, because only the best passer went to 2024/25. Do not check starter after that result. No winner.

## Rules gaps

- Full Opta BPS table (goals, assists, playing time, and the rest) is not in the 20 Jul 2026 change note. Only the 2026/27 deltas are coded.
- Chip-aware backtest: Wildcard keeps banked transfers; Free Hit squad reverts; Bench Boost adds the bench; Triple Captain is ×3. No season runner calls `ChipWallet` yet.
- Rescoring historical seasons under 2026/27 BPS needs CBI, save location, and big-chance saves. Do not overwrite Vaastav `total_points` until those fields exist.
