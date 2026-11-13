# Card comparison

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Each training card is cut after 20%, 40%, 60%, or 80% of its fixtures,
chosen at random, starting from gameweek 1. The test scores all four cuts.
Every club and every row is included.

The reader is squad vision on the square card. The flashcard is the
only feature input. Each column shows the club symbol, the opponent
symbol, home or away, the handicap line, and the 2.5 total. A faint
grid outlines every square. The
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
| 20% | goals | 730 | 501 | 106 | 220 | 197 |
| 20% | assists | 707 | 535 | 91 | 180 | 171 |
| 40% | goals | 544 | 372 | 82 | 172 | 152 |
| 40% | assists | 531 | 393 | 65 | 131 | 124 |
| 60% | goals | 360 | 221 | 57 | 107 | 95 |
| 60% | assists | 351 | 250 | 47 | 86 | 83 |
| 80% | goals | 199 | 111 | 27 | 61 | 49 |
| 80% | assists | 192 | 105 | 18 | 36 | 32 |

Goal and assist bars by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/46, MID 394/270, FWD 233/185.
assists: actual/painted GKP 4/0, DEF 181/77, MID 434/378, FWD 88/80.
goal chance: GKP 0.006, DEF 0.012, MID 0.039, FWD 0.099.
assist chance: GKP 0.007, DEF 0.016, MID 0.053, FWD 0.044.

The loss scores each hidden square and each row's own hidden bar count.
A matching card total with the bars on the empty rows is an error.
The paint keeps each row's count on that row. The flashcard is the
only feature input.

Wall clock: 107 seconds on CPU.

