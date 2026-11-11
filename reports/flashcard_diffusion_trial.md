# Diffusion painter

The training set is the rendered card. Each square is one cell.
A future goal or assist square is gray. Those squares are noised,
and the painter restores the clean picture in 32 steps. The visible
card, including the minute stack, stays in place at every step.
A bar is counted where the restored color is closer to the goal or
assist color than to white or to the gray blank. One finished card
is one draw. The other squad is not an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 943 | 181 | 324 | 365 |
| 20% | assists | 707 | 276 | 44 | 97 | 86 |
| 40% | goals | 544 | 689 | 138 | 243 | 285 |
| 40% | assists | 531 | 221 | 35 | 73 | 71 |
| 60% | goals | 360 | 440 | 92 | 162 | 186 |
| 60% | assists | 351 | 143 | 21 | 43 | 40 |
| 80% | goals | 199 | 225 | 53 | 88 | 95 |
| 80% | assists | 192 | 75 | 12 | 21 | 21 |

The rank is the hit rate times the bar-count match. The gate is the
21:47 UK full painter. The remainder score drew a count and then
picked weeks. The picture score matched the colors in one pass and
painted no bars. This run does not replace the gate.

| revealed | channel | actual bars | gate bars | restored bars | gate hits | restored hits | gate score | remainder score | picture score | restored score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 759 | 943 | 192 | 181 | 0.248 | 0.134 | 0.000 | 0.168 |
| 20% | assists | 707 | 692 | 276 | 100 | 44 | 0.140 | 0.088 | 0.000 | 0.035 |
| 40% | goals | 544 | 589 | 689 | 147 | 138 | 0.240 | 0.127 | 0.000 | 0.177 |
| 40% | assists | 531 | 535 | 221 | 78 | 35 | 0.145 | 0.085 | 0.000 | 0.039 |
| 60% | goals | 360 | 384 | 440 | 94 | 92 | 0.237 | 0.131 | 0.000 | 0.188 |
| 60% | assists | 351 | 348 | 143 | 55 | 21 | 0.156 | 0.065 | 0.000 | 0.035 |
| 80% | goals | 199 | 204 | 225 | 55 | 53 | 0.266 | 0.092 | 0.000 | 0.221 |
| 80% | assists | 192 | 185 | 75 | 36 | 12 | 0.184 | 0.100 | 0.000 | 0.035 |

The busiest string is the most goal appearances one player has in the
hidden window. The typical string is the median of that count across
clubs. The regular share is the fraction of a regular's played weeks
that contain a goal. A regular plays at least half the hidden weeks.

| revealed | actual busiest | painted busiest | actual typical | painted typical | actual regular share | painted regular share |
|---|---:|---:|---:|---:|---:|---:|
| 20% | 13 | 22 | 8 | 13 | 9% | 12% |
| 40% | 10 | 17 | 6 | 10 | 9% | 11% |
| 60% | 8 | 12 | 5 | 6 | 9% | 11% |
| 80% | 5 | 6 | 3 | 4 | 9% | 10% |

A goal is paired when another player with minutes in that gameweek
has an assist. A run is a goal appearance with another goal by the
same player on a neighboring hidden fixture.

| revealed | actual goals with an assist beside them | painted |
|---|---|---|
| 20% | 703/730 (96%) | 443/943 (47%) |
| 40% | 525/544 (97%) | 341/689 (49%) |
| 60% | 346/360 (96%) | 228/440 (52%) |
| 80% | 191/199 (96%) | 106/225 (47%) |

| revealed | actual goals in a run | painted |
|---|---|---|
| 20% | 224/730 (31%) | 384/943 (41%) |
| 40% | 167/544 (31%) | 293/689 (43%) |
| 60% | 98/360 (27%) | 190/440 (43%) |
| 80% | 52/199 (26%) | 79/225 (35%) |

With the future event squares left gray, the share of appearances
that receive an event square. The squares are listed from the bottom.

| source | minutes | 1st square | 2nd square | 3rd square | any square |
|---|---|---:|---:|---:|---:|
| 20% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual goals | played | 0.081 | 0.008 | 0.001 | 0.081 |
| 20% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual assists | played | 0.078 | 0.005 | 0.000 | 0.078 |
| 20% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted goals | played | 0.104 | 0.000 | 0.000 | 0.104 |
| 20% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted assists | played | 0.031 | 0.000 | 0.000 | 0.031 |
| 40% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual goals | played | 0.079 | 0.007 | 0.001 | 0.079 |
| 40% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual assists | played | 0.077 | 0.004 | 0.000 | 0.077 |
| 40% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted goals | played | 0.100 | 0.000 | 0.000 | 0.100 |
| 40% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted assists | played | 0.032 | 0.000 | 0.000 | 0.032 |
| 60% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual goals | played | 0.080 | 0.007 | 0.001 | 0.080 |
| 60% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual assists | played | 0.078 | 0.004 | 0.000 | 0.078 |
| 60% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted goals | played | 0.097 | 0.000 | 0.000 | 0.097 |
| 60% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted assists | played | 0.032 | 0.000 | 0.000 | 0.032 |
| 80% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual goals | played | 0.082 | 0.008 | 0.000 | 0.082 |
| 80% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual assists | played | 0.079 | 0.005 | 0.000 | 0.079 |
| 80% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted goals | played | 0.093 | 0.000 | 0.000 | 0.093 |
| 80% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted assists | played | 0.031 | 0.000 | 0.000 | 0.031 |

A painted stack skips a lower square when a higher square is filled
and a square below it is empty.

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 943 | 0 | 276 | 0 |
| 40% | 689 | 0 | 221 | 0 |
| 60% | 440 | 0 | 143 | 0 |
| 80% | 225 | 0 | 75 | 0 |
