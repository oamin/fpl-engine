# Card comparison

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Each training card is cut after 20%, 40%, 60%, or 80% of its fixtures,
chosen at random, starting from gameweek 1. The test scores all four cuts.
Every club and every row is included.

The reader is squad vision on the square card. The flashcard is the
only feature input. Each column shows the club symbol, the opponent
symbol, home or away, then sixteen solid handicap squares scaled from
giving 1.5 to receiving 1.5, and eight solid 2.5 squares. The
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
| 20% | goals | 730 | 635 | 129 | 268 | 256 |
| 20% | assists | 707 | 568 | 95 | 190 | 177 |
| 40% | goals | 544 | 479 | 101 | 211 | 201 |
| 40% | assists | 531 | 412 | 69 | 137 | 126 |
| 60% | goals | 360 | 301 | 67 | 137 | 123 |
| 60% | assists | 351 | 255 | 46 | 86 | 80 |
| 80% | goals | 199 | 143 | 36 | 74 | 60 |
| 80% | assists | 192 | 108 | 19 | 37 | 33 |

Goal and assist bars by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/65, MID 394/332, FWD 233/238.
assists: actual/painted GKP 4/0, DEF 181/91, MID 434/386, FWD 88/91.
goal chance: GKP 0.005, DEF 0.014, MID 0.048, FWD 0.127.
assist chance: GKP 0.005, DEF 0.018, MID 0.053, FWD 0.050.

The loss scores each hidden square and each row's own hidden bar count.
A matching card total with the bars on the empty rows is an error.
The paint keeps each row's count on that row. The flashcard is the
only feature input.

Wall clock: 73 seconds on CPU.

