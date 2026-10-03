# Stage 36 — scheduled minutes, across seasons

The +34 bar stays the bar for replacing `score_xp`. It is about one point a week, and a single season can clear it and then give the points back the next year. This test asks whether the scheduled-minutes score keeps its sign. The rule was locked before these seasons were read. 2025/26 is the run already on file: fast XI −4, free-transfer climb +9.

A season whose fast XI is 100 or more behind is not climbed, and that kill makes the column unstable. Stable means the worst climbed season loses by less than 34, at least three climbed seasons are level or ahead, at least three seasons were climbed, and the gaps sum to more than zero. Stable does not replace `score_xp`.

**UNSTABLE**

| season | fast gap | transfer gap | weeks ahead or level | worst week | weekly sd | scores moved |
|---|---:|---:|---:|---:|---:|---:|
| 2025-26 | -4 | +9 | 62% | -19 | 7.7 | 15% |
| 2022-23 | +76 | -78 | 50% | -21 | 7.5 | 14% |
| 2023-24 | +51 | +146 | 76% | -21 | 13.4 | 15% |
| 2024-25 | +10 | -44 | 53% | -24 | 10.5 | 14% |

Weeks ahead or level, the worst week, and the weekly standard deviation are the free-transfer climb. Scores moved is the share of eligible rows in Gameweeks 5–38 where the two scores differ by more than 0.5. Those three columns are diagnostics. They are not a second gate.

Gemini reviewed the table. The verdict stays unstable, and the column stays parked. The +146 in 2023/24 is three weeks (+25, +36, +45). 2022/23 never has a week better than +9 and finishes 78 behind, with 64 hit points against 56. 2023/24 takes 40 hit points against 28. 2024/25 is level on hits and 44 behind. About 15% of eligible rows move by more than half a point, and the season totals still swing from −78 to +146. The +34 bar was not hiding a small edge that kept its sign. The transfer gap itself changes sign. `pass_margin` stays 34. `score_xp` stays the published score.

The fast XI is the quieter of the two tests: +76, +51, +10, and −4. No season was killed. The ranking among players who played does not blow up. The squad path does.

