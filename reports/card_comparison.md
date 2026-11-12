# Card comparison

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Each training card is cut after 20%, 40%, 60%, or 80% of its fixtures,
chosen at random, starting from gameweek 1. The test scores all four cuts.
Every club and every row is included.

The reader is squad vision on the square card. The flashcard is the
only feature input. Each column shows the club symbol, the opponent
symbol, home or away, then sixteen solid handicap squares. The
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
| 20% | goals | 730 | 449 | 97 | 202 | 174 |
| 20% | assists | 707 | 445 | 79 | 159 | 144 |
| 40% | goals | 544 | 313 | 71 | 152 | 131 |
| 40% | assists | 531 | 310 | 58 | 108 | 101 |
| 60% | goals | 360 | 188 | 53 | 105 | 88 |
| 60% | assists | 351 | 181 | 30 | 62 | 58 |
| 80% | goals | 199 | 93 | 25 | 56 | 45 |
| 80% | assists | 192 | 82 | 16 | 32 | 28 |

Goal and assist bars by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/45, MID 394/248, FWD 233/156.
assists: actual/painted GKP 4/0, DEF 181/75, MID 434/297, FWD 88/73.
goal chance: GKP 0.006, DEF 0.012, MID 0.035, FWD 0.085.
assist chance: GKP 0.006, DEF 0.015, MID 0.042, FWD 0.039.

The loss scores each hidden square and each row's own hidden bar count.
A matching card total with the bars on the empty rows is an error.
The paint keeps each row's count on that row. The flashcard is the
only feature input.

Wall clock: 71 seconds on CPU.

