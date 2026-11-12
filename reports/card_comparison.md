# Card comparison

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Each training card is cut after 20%, 40%, 60%, or 80% of its fixtures,
chosen at random, starting from gameweek 1. The test scores all four cuts.
Every club and every row is included.

The reader is squad vision on the square card. The flashcard is the
only feature input. Each column shows the club symbol, the opponent
symbol, home or away, then sixteen 1X2 squares. The
other squad is not an input. Concede, regulars, own history,
opponent history, and the team scoring rate are absent. Minutes,
position, club, and opponent are not passed again as numbers.
One reader trains on two views, and both losses update the same
weights. The horizon view reveals the masked remainder in one shot.
The column view keeps the true past, shows one next week, and scores
only that week. Each loss scores each hidden square and each row's
own hidden bar count. The paint keeps that count on the same row.
A row holds three squares, and that is the only cap.

Horizon, one shot. The test scores all four cuts of this view.

A bar matches on the same row only. Within one gameweek includes the same
appearance and either neighboring gameweek. The numbers are totals.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 482 | 103 | 204 | 174 |
| 20% | assists | 707 | 455 | 75 | 138 | 129 |
| 40% | goals | 544 | 360 | 78 | 160 | 137 |
| 40% | assists | 531 | 340 | 57 | 101 | 94 |
| 60% | goals | 360 | 200 | 50 | 93 | 77 |
| 60% | assists | 351 | 189 | 36 | 58 | 54 |
| 80% | goals | 199 | 89 | 23 | 42 | 36 |
| 80% | assists | 192 | 82 | 14 | 29 | 25 |

Goal and assist bars by position at the 20% cut.

goals: actual/painted GKP 0/21, DEF 103/67, MID 394/217, FWD 233/177.
assists: actual/painted GKP 4/16, DEF 181/87, MID 434/270, FWD 88/82.
goal chance: GKP 0.021, DEF 0.014, MID 0.031, FWD 0.098.
assist chance: GKP 0.016, DEF 0.017, MID 0.039, FWD 0.043.

Next column. The same weights see the true past and one next week.
Each hidden week below is painted from that week's own column.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 0 | 0 | 0 | 0 |
| 20% | assists | 707 | 0 | 0 | 0 | 0 |
| 40% | goals | 544 | 0 | 0 | 0 | 0 |
| 40% | assists | 531 | 0 | 0 | 0 | 0 |
| 60% | goals | 360 | 0 | 0 | 0 | 0 |
| 60% | assists | 351 | 0 | 0 | 0 | 0 |
| 80% | goals | 199 | 0 | 0 | 0 | 0 |
| 80% | assists | 192 | 0 | 0 | 0 | 0 |

Mean chance of a bar on the guessed column, on a real bar and on a blank: goals 0.106 / 0.041, assists 0.075 / 0.035.

Both losses update the same weights. Each scores a hidden square and
each row's own hidden bar count.
A matching card total with the bars on the empty rows is an error.
The paint keeps each row's count on that row. The flashcard is the
only feature input.

Wall clock: 149 seconds on CPU.

