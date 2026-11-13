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
Two readers see that card, and their weights stay separate. Goals use
fixture intensity: the match header owns that week's expected bar
total, then the squad shares it. Assists keep a four-way head. The
paint keeps each row's horizon count on that row and moves the bars
toward the weeks the column reader scores highest. A row whose
horizon count is zero stays empty. A row holds three squares, and
that is the only cap.

A bar matches on the same row only. Within one gameweek includes the same
appearance and either neighboring gameweek. The numbers are totals.

Horizon rows. This paint uses the horizon reader alone.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 767 | 150 | 293 | 321 |
| 20% | assists | 707 | 643 | 100 | 194 | 178 |
| 40% | goals | 544 | 590 | 111 | 219 | 242 |
| 40% | assists | 531 | 522 | 75 | 159 | 147 |
| 60% | goals | 360 | 369 | 77 | 158 | 158 |
| 60% | assists | 351 | 351 | 52 | 107 | 95 |
| 80% | goals | 199 | 172 | 37 | 86 | 78 |
| 80% | assists | 192 | 144 | 29 | 51 | 43 |

Horizon rows, by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/96, MID 394/377, FWD 233/294.
assists: actual/painted GKP 4/21, DEF 181/134, MID 434/361, FWD 88/127.
goal chance: GKP 0.005, DEF 0.020, MID 0.054, FWD 0.159.
assist chance: GKP 0.021, DEF 0.026, MID 0.052, FWD 0.069.

Column weeks. This paint uses the column reader alone. Each row
keeps that reader's own count, and each week is one count down the
squad.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 771 | 103 | 229 | 223 |
| 20% | assists | 707 | 649 | 85 | 165 | 150 |
| 40% | goals | 544 | 571 | 79 | 165 | 160 |
| 40% | assists | 531 | 496 | 61 | 121 | 110 |
| 60% | goals | 360 | 388 | 51 | 120 | 119 |
| 60% | assists | 351 | 319 | 40 | 85 | 74 |
| 80% | goals | 199 | 161 | 26 | 68 | 57 |
| 80% | assists | 192 | 126 | 24 | 41 | 35 |

Column weeks, by position at the 20% cut.

goals: actual/painted GKP 0/70, DEF 103/157, MID 394/336, FWD 233/208.
assists: actual/painted GKP 4/80, DEF 181/131, MID 434/324, FWD 88/114.

Fitted weeks. Each row keeps its horizon count. The bars move
toward the weeks the column reader scores highest.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 767 | 158 | 291 | 319 |
| 20% | assists | 707 | 643 | 97 | 184 | 168 |
| 40% | goals | 544 | 590 | 110 | 218 | 236 |
| 40% | assists | 531 | 522 | 71 | 134 | 124 |
| 60% | goals | 360 | 369 | 73 | 153 | 156 |
| 60% | assists | 351 | 351 | 49 | 100 | 88 |
| 80% | goals | 199 | 172 | 41 | 87 | 78 |
| 80% | assists | 192 | 144 | 24 | 43 | 37 |

Fitted weeks, by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/96, MID 394/377, FWD 233/294.
assists: actual/painted GKP 4/21, DEF 181/134, MID 434/361, FWD 88/127.

Column reader at the 20% cut, mean predicted/actual count and correlation: goals 1.30/1.22 (0.15), assists 1.13/1.18 (0.22).

The two readers keep separate weights. Row and week counts are
Poisson rates, so the volume is absolute. A row whose horizon
count is zero stays empty. A matching card total with the bars
on the empty rows is an error. The flashcard is the only feature
input. The picture, the layout, and both state dicts are stored
in `reports/cards/current/`. The table above is scored from those
files.

Wall clock: 142 seconds on CPU.

