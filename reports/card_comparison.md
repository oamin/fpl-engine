# Card comparison

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Each training card is cut after 20%, 40%, 60%, or 80% of its fixtures,
chosen at random, starting from gameweek 1. The test scores all four cuts.
Every club and every row is included.

The reader is squad vision on the square card. The flashcard is the
only feature input. The opponent is one square, a club symbol. The
other squad is not an input. Concede, regulars, own history,
opponent history, and the team scoring rate are absent. Minutes,
position, club, and opponent are not passed again as numbers. Each
loss scores each hidden square and each row's own hidden bar count.
The paint keeps that count on the same row. A row holds three
squares, and that is the only cap.

A bar matches on the same row only. Within one gameweek includes the same
appearance and either neighboring gameweek. The numbers are totals.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 798 | 169 | 322 | 323 |
| 20% | assists | 707 | 664 | 107 | 215 | 203 |
| 40% | goals | 544 | 594 | 129 | 241 | 248 |
| 40% | assists | 531 | 498 | 80 | 159 | 147 |
| 60% | goals | 360 | 368 | 88 | 153 | 149 |
| 60% | assists | 351 | 302 | 53 | 100 | 96 |
| 80% | goals | 199 | 184 | 43 | 84 | 74 |
| 80% | assists | 192 | 140 | 30 | 48 | 43 |

Goal and assist bars by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/71, MID 394/422, FWD 233/305.
assists: actual/painted GKP 4/0, DEF 181/90, MID 434/452, FWD 88/122.
goal chance: GKP 0.007, DEF 0.014, MID 0.061, FWD 0.165.
assist chance: GKP 0.007, DEF 0.018, MID 0.063, FWD 0.065.

The loss scores each hidden square and each row's own hidden bar count.
A matching card total with the bars on the empty rows is an error.
The paint keeps each row's count on that row. The flashcard is the
only feature input.

Wall clock: 73 seconds on CPU.

