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
Two readers see that card, and their weights stay separate. The
horizon reader scores each hidden square and a Poisson rate on each
row's hidden bar count. The column reader scores a Poisson rate on
each hidden week's bar count. Absolute rates pin the volume. The
paint keeps each row's horizon count on that row and moves the bars
toward the weeks the column reader scores highest. A row whose
horizon count is zero stays empty. A row holds three squares, and
that is the only cap.

A bar matches on the same row only. Within one gameweek includes the same
appearance and either neighboring gameweek. The numbers are totals.

Horizon rows. This paint uses the horizon reader alone.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 907 | 170 | 341 | 321 |
| 20% | assists | 707 | 884 | 144 | 265 | 256 |
| 40% | goals | 544 | 680 | 125 | 256 | 237 |
| 40% | assists | 531 | 671 | 111 | 203 | 193 |
| 60% | goals | 360 | 445 | 92 | 170 | 157 |
| 60% | assists | 351 | 430 | 75 | 124 | 121 |
| 80% | goals | 199 | 206 | 47 | 86 | 77 |
| 80% | assists | 192 | 219 | 40 | 72 | 64 |

Horizon rows, by position at the 20% cut.

goals: actual/painted GKP 0/2, DEF 103/132, MID 394/504, FWD 233/269.
assists: actual/painted GKP 4/5, DEF 181/198, MID 434/544, FWD 88/137.
goal chance: GKP 0.008, DEF 0.026, MID 0.069, FWD 0.146.
assist chance: GKP 0.009, DEF 0.037, MID 0.076, FWD 0.073.

Column weeks. This paint uses the column reader alone. Each row
keeps that reader's own count, and each week is one count down the
squad.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 732 | 119 | 241 | 219 |
| 20% | assists | 707 | 736 | 104 | 206 | 190 |
| 40% | goals | 544 | 564 | 95 | 174 | 167 |
| 40% | assists | 531 | 548 | 73 | 136 | 127 |
| 60% | goals | 360 | 357 | 65 | 129 | 114 |
| 60% | assists | 351 | 365 | 56 | 103 | 94 |
| 80% | goals | 199 | 162 | 35 | 71 | 59 |
| 80% | assists | 192 | 164 | 24 | 42 | 38 |

Column weeks, by position at the 20% cut.

goals: actual/painted GKP 0/60, DEF 103/116, MID 394/350, FWD 233/206.
assists: actual/painted GKP 4/42, DEF 181/170, MID 434/439, FWD 88/85.

Fitted weeks. Each row keeps its horizon count. The bars move
toward the weeks the column reader scores highest.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 907 | 168 | 324 | 321 |
| 20% | assists | 707 | 884 | 135 | 251 | 239 |
| 40% | goals | 544 | 680 | 126 | 237 | 234 |
| 40% | assists | 531 | 671 | 94 | 175 | 167 |
| 60% | goals | 360 | 445 | 85 | 159 | 146 |
| 60% | assists | 351 | 430 | 62 | 107 | 103 |
| 80% | goals | 199 | 206 | 44 | 88 | 74 |
| 80% | assists | 192 | 219 | 35 | 64 | 56 |

Fitted weeks, by position at the 20% cut.

goals: actual/painted GKP 0/2, DEF 103/132, MID 394/504, FWD 233/269.
assists: actual/painted GKP 4/5, DEF 181/198, MID 434/544, FWD 88/137.

Column reader at the 20% cut, mean predicted/actual count and correlation: goals 1.26/1.22 (0.18), assists 1.25/1.18 (0.24).

The two readers keep separate weights. Row and week counts are
Poisson rates, so the volume is absolute. A row whose horizon
count is zero stays empty. A matching card total with the bars
on the empty rows is an error. The flashcard is the only feature
input. The picture, the layout, and both state dicts are stored
in `reports/cards/current/`. The table above is scored from those
files.

Wall clock: 152 seconds on CPU.

