# Column flashcard learner

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
The stepped square card is the only input. The model pools the next
column into one object and paints that column's event strip in one
pass. The loss is the strip color error. There is no per-cell class
head, no Poisson rate, no share, and no argmax paint rule. Bars are
read from ink afterwards.

Diagrams: `column_picture_flow.png` (processing),
`column_picture_teach.png` (one training example).

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 3755 | 511 | 573 | 1125 |
| 20% | assists | 707 | 3750 | 424 | 469 | 950 |
| 40% | goals | 544 | 2863 | 384 | 429 | 840 |
| 40% | assists | 531 | 2861 | 316 | 351 | 700 |
| 60% | goals | 360 | 1849 | 249 | 282 | 541 |
| 60% | assists | 351 | 1847 | 209 | 232 | 452 |
| 80% | goals | 199 | 975 | 136 | 152 | 274 |
| 80% | assists | 192 | 975 | 115 | 128 | 233 |

Column flashcard learner, by position at the 20% cut.

goals: actual/painted GKP 0/588, DEF 103/0, MID 394/2529, FWD 233/638.
assists: actual/painted GKP 4/588, DEF 181/0, MID 434/2525, FWD 88/637.

Core idea: the flashcard is one picture. The column is one object.
Do not replace this with cell posteriors, volume caps, or share rules.

Wall clock: 57 seconds on CPU.

