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
horizon reader scores each hidden square and each row's own hidden
bar count. The column reader scores each hidden week's bar count.
The paint keeps each row's horizon count on that row and moves the
bars toward the weeks the column reader scores highest. A row whose
horizon count is zero stays empty. A row holds three squares, and
that is the only cap.

A bar matches on the same row only. Within one gameweek includes the same
appearance and either neighboring gameweek. The numbers are totals.

Horizon rows. This paint uses the horizon reader alone.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 593 | 139 | 268 | 249 |
| 20% | assists | 707 | 480 | 88 | 168 | 155 |
| 40% | goals | 544 | 434 | 105 | 203 | 185 |
| 40% | assists | 531 | 350 | 58 | 111 | 106 |
| 60% | goals | 360 | 272 | 75 | 137 | 120 |
| 60% | assists | 351 | 210 | 40 | 71 | 67 |
| 80% | goals | 199 | 126 | 37 | 70 | 58 |
| 80% | assists | 192 | 92 | 23 | 37 | 33 |

Horizon rows, by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/53, MID 394/312, FWD 233/228.
assists: actual/painted GKP 4/0, DEF 181/65, MID 434/336, FWD 88/79.
goal chance: GKP 0.006, DEF 0.012, MID 0.044, FWD 0.126.
assist chance: GKP 0.006, DEF 0.014, MID 0.047, FWD 0.045.

Fitted weeks. Each row keeps its horizon count. The bars move
toward the weeks the column reader scores highest.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 593 | 138 | 252 | 239 |
| 20% | assists | 707 | 480 | 80 | 155 | 141 |
| 40% | goals | 544 | 434 | 101 | 200 | 184 |
| 40% | assists | 531 | 350 | 55 | 102 | 92 |
| 60% | goals | 360 | 272 | 65 | 124 | 110 |
| 60% | assists | 351 | 210 | 32 | 61 | 55 |
| 80% | goals | 199 | 126 | 36 | 70 | 60 |
| 80% | assists | 192 | 92 | 17 | 30 | 25 |

Fitted weeks, by position at the 20% cut.

goals: actual/painted GKP 0/0, DEF 103/53, MID 394/312, FWD 233/228.
assists: actual/painted GKP 4/0, DEF 181/65, MID 434/336, FWD 88/79.

Column reader at the 20% cut, mean predicted/actual count and correlation: goals 0.91/1.22 (0.21), assists 0.86/1.18 (0.23).

The two readers keep separate weights. A row whose horizon count
is zero stays empty. A matching card total with the bars on the
empty rows is an error. The flashcard is the only feature input.
The picture, the layout, and both state dicts are stored in
`reports/cards/current/`. The table above is scored from those files.

Wall clock: 149 seconds on CPU.

