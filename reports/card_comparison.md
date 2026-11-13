# Card comparison

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Each training card is cut after 20%, 40%, 60%, or 80% of its fixtures,
chosen at random, starting from gameweek 1. The test scores all four cuts.
Every club and every row is included.

The reader is squad vision on the square card. The flashcard is the
only feature input. Each column shows the club symbol, the opponent
symbol, home or away, the handicap line, and the 2.5 total. The
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
| 20% | goals | 730 | 538 | 108 | 227 | 205 |
| 20% | assists | 707 | 521 | 87 | 176 | 165 |
| 40% | goals | 544 | 404 | 87 | 187 | 166 |
| 40% | assists | 531 | 386 | 66 | 129 | 121 |
| 60% | goals | 360 | 243 | 56 | 112 | 101 |
| 60% | assists | 351 | 247 | 46 | 84 | 81 |
| 80% | goals | 199 | 114 | 29 | 62 | 51 |
| 80% | assists | 192 | 96 | 19 | 35 | 31 |

Goal and assist bars by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/62, MID 394/277, FWD 233/199.
assists: actual/painted GKP 4/0, DEF 181/94, MID 434/347, FWD 88/80.
goal chance: GKP 0.006, DEF 0.014, MID 0.039, FWD 0.108.
assist chance: GKP 0.006, DEF 0.019, MID 0.049, FWD 0.043.

The loss scores each hidden square and each row's own hidden bar count.
A matching card total with the bars on the empty rows is an error.
The paint keeps each row's count on that row. The flashcard is the
only feature input.

Wall clock: 72 seconds on CPU.

