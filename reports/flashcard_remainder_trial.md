# Remainder painter

The training set is the rendered card. Each square is one cell.
A future goal or assist square is gray. The painter reads the whole
card once and draws one code for the gray remainder. The code sets a
player pattern and a week pattern, and a square is painted where they
meet on the visible card. Odds, venue, the opponent mark, the minute
stack, and the position color stay on the picture, including on a
future gameweek. The code is one of the remainder patterns learned
from the training pictures. Training matches the squares and each
player's own run of hidden weeks. The draw keeps that many weeks
for each player and paints those weeks. Each finished
card is one draw at temperature 1. The other squad is not an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 884 | 131 | 272 | 257 |
| 20% | assists | 707 | 808 | 76 | 169 | 174 |
| 40% | goals | 544 | 619 | 84 | 186 | 181 |
| 40% | assists | 531 | 596 | 54 | 128 | 128 |
| 60% | goals | 360 | 439 | 64 | 125 | 125 |
| 60% | assists | 351 | 402 | 28 | 84 | 76 |
| 80% | goals | 199 | 231 | 23 | 61 | 55 |
| 80% | assists | 192 | 222 | 24 | 46 | 44 |

One code is drawn for the whole remainder. It sets which players and
which weeks carry the events, and a square is painted where the two
meet. The stored
gate is the 21:47 UK full painter. This run does not replace it.

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 884 | 131 | 192 | -61 | 272 | 256 | +16 |
| 20% | assists | 707 | 808 | 76 | 100 | -24 | 169 | 161 | +8 |
| 40% | goals | 544 | 619 | 84 | 147 | -63 | 186 | 200 | -14 |
| 40% | assists | 531 | 596 | 54 | 78 | -24 | 128 | 119 | +9 |
| 60% | goals | 360 | 439 | 64 | 94 | -30 | 125 | 134 | -9 |
| 60% | assists | 351 | 402 | 28 | 55 | -27 | 84 | 84 | +0 |
| 80% | goals | 199 | 231 | 23 | 55 | -32 | 61 | 73 | -12 |
| 80% | assists | 192 | 222 | 24 | 36 | -12 | 46 | 51 | -5 |

The busiest string is the most goal appearances one player has in the
hidden window. The typical string is the median of that count across
clubs. The regular share is the fraction of a regular's played weeks
that contain a goal. A regular plays at least half the hidden weeks.

| revealed | actual busiest | painted busiest | actual typical | painted typical | actual regular share | painted regular share |
|---|---:|---:|---:|---:|---:|---:|
| 20% | 13 | 16 | 8 | 8 | 9% | 10% |
| 40% | 10 | 11 | 6 | 6 | 9% | 9% |
| 60% | 8 | 9 | 5 | 5 | 9% | 10% |
| 80% | 5 | 4 | 3 | 2 | 9% | 9% |

A goal is paired when another player with minutes in that gameweek
has an assist. A run is a goal appearance with another goal by the
same player on a neighboring hidden fixture.

| revealed | actual goals with an assist beside them | painted |
|---|---|---|
| 20% | 703/730 (96%) | 586/838 (70%) |
| 40% | 525/544 (97%) | 394/584 (67%) |
| 60% | 346/360 (96%) | 294/425 (69%) |
| 80% | 191/199 (96%) | 154/206 (75%) |

| revealed | actual goals in a run | painted |
|---|---|---|
| 20% | 224/730 (31%) | 206/884 (23%) |
| 40% | 167/544 (31%) | 146/619 (24%) |
| 60% | 98/360 (27%) | 120/439 (27%) |
| 80% | 52/199 (26%) | 45/231 (19%) |

With the future event squares left gray, the share of appearances
that receive an event square. The squares are listed from the bottom.

| source | minutes | 1st square | 2nd square | 3rd square | any square |
|---|---|---:|---:|---:|---:|
| 20% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual goals | played | 0.081 | 0.008 | 0.001 | 0.081 |
| 20% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual assists | played | 0.078 | 0.005 | 0.000 | 0.078 |
| 20% painted goals | zero | 0.007 | 0.000 | 0.000 | 0.007 |
| 20% painted goals | played | 0.093 | 0.003 | 0.000 | 0.093 |
| 20% painted assists | zero | 0.008 | 0.000 | 0.000 | 0.008 |
| 20% painted assists | played | 0.084 | 0.002 | 0.000 | 0.084 |
| 40% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual goals | played | 0.079 | 0.007 | 0.001 | 0.079 |
| 40% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual assists | played | 0.077 | 0.004 | 0.000 | 0.077 |
| 40% painted goals | zero | 0.007 | 0.000 | 0.000 | 0.007 |
| 40% painted goals | played | 0.085 | 0.003 | 0.000 | 0.085 |
| 40% painted assists | zero | 0.008 | 0.000 | 0.000 | 0.008 |
| 40% painted assists | played | 0.081 | 0.002 | 0.000 | 0.081 |
| 60% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual goals | played | 0.080 | 0.007 | 0.001 | 0.080 |
| 60% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual assists | played | 0.078 | 0.004 | 0.000 | 0.078 |
| 60% painted goals | zero | 0.004 | 0.000 | 0.000 | 0.004 |
| 60% painted goals | played | 0.094 | 0.002 | 0.000 | 0.094 |
| 60% painted assists | zero | 0.007 | 0.000 | 0.000 | 0.007 |
| 60% painted assists | played | 0.083 | 0.001 | 0.000 | 0.083 |
| 80% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual goals | played | 0.082 | 0.008 | 0.000 | 0.082 |
| 80% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual assists | played | 0.079 | 0.005 | 0.000 | 0.079 |
| 80% painted goals | zero | 0.013 | 0.000 | 0.000 | 0.013 |
| 80% painted goals | played | 0.085 | 0.002 | 0.000 | 0.085 |
| 80% painted assists | zero | 0.004 | 0.000 | 0.000 | 0.004 |
| 80% painted assists | played | 0.089 | 0.001 | 0.000 | 0.089 |

A painted stack skips a lower square when a higher square is filled
and a square below it is empty.

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 884 | 0 | 808 | 0 |
| 40% | 619 | 0 | 596 | 0 |
| 60% | 439 | 0 | 402 | 0 |
| 80% | 231 | 0 | 222 | 0 |
