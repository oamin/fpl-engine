# Stepped column one-shot

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
The reader is the locked fixture-intensity head. Training shows the
true past and one next week. That week is one object: a Poisson rate
for how many bars, then a share over the squad for who. Per-cell
Bernoulli is not used. Paint walks every gameweek the same way.
There is no horizon reader and no fitted mix in this table.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 610 | 146 | 189 | 300 |
| 20% | assists | 707 | 599 | 90 | 142 | 181 |
| 40% | goals | 544 | 470 | 109 | 140 | 228 |
| 40% | assists | 531 | 459 | 58 | 98 | 123 |
| 60% | goals | 360 | 310 | 68 | 90 | 143 |
| 60% | assists | 351 | 299 | 37 | 62 | 75 |
| 80% | goals | 199 | 168 | 42 | 55 | 80 |
| 80% | assists | 192 | 160 | 24 | 39 | 43 |

Stepped column one-shot, by position at the 20% cut.

goals: actual/painted GKP 0/89, DEF 103/0, MID 394/58, FWD 233/463.
assists: actual/painted GKP 4/7, DEF 181/0, MID 434/523, FWD 88/69.
goal chance on bar 0.140, off bar 0.051.
assist chance on bar 0.074, off bar 0.034.

Each column is scored as a whole. Rows compete for the week's
mass, so a position that never scores cannot keep the bars. The
flashcard is the only feature input.

Wall clock: 83 seconds on CPU.

