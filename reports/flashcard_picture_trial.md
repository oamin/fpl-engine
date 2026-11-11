# Picture painter

The training set is the rendered card. Each square is one cell.
A future goal or assist square is gray. The painter reads the whole
card once and writes a color on every future event square. Training
is the average absolute difference between that picture and the real
card. A bar is counted where the painted color is closer to the goal
or assist color than to white or to the gray blank. The other squad
is not an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

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

The rank is the hit rate times the bar-count match. The gate is the
21:47 UK full painter. The remainder score is the previous painter,
which drew a count and then picked weeks. This run does not replace
the gate.

| revealed | channel | actual bars | gate bars | picture bars | gate hits | picture hits | gate score | remainder score | picture score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 759 | 0 | 192 | 0 | 0.248 | 0.134 | 0.000 |
| 20% | assists | 707 | 692 | 0 | 100 | 0 | 0.140 | 0.088 | 0.000 |
| 40% | goals | 544 | 589 | 0 | 147 | 0 | 0.240 | 0.127 | 0.000 |
| 40% | assists | 531 | 535 | 0 | 78 | 0 | 0.145 | 0.085 | 0.000 |
| 60% | goals | 360 | 384 | 0 | 94 | 0 | 0.237 | 0.131 | 0.000 |
| 60% | assists | 351 | 348 | 0 | 55 | 0 | 0.156 | 0.065 | 0.000 |
| 80% | goals | 199 | 204 | 0 | 55 | 0 | 0.266 | 0.092 | 0.000 |
| 80% | assists | 192 | 185 | 0 | 36 | 0 | 0.184 | 0.100 | 0.000 |

Color error is the average absolute difference on the gray squares,
on the same 0 to 1 scale as the picture.

| revealed | color error |
|---|---:|
| 20% | 0.0176 |
| 40% | 0.0173 |
| 60% | 0.0173 |
| 80% | 0.0176 |

The busiest string is the most goal appearances one player has in the
hidden window. The typical string is the median of that count across
clubs. The regular share is the fraction of a regular's played weeks
that contain a goal. A regular plays at least half the hidden weeks.

| revealed | actual busiest | painted busiest | actual typical | painted typical | actual regular share | painted regular share |
|---|---:|---:|---:|---:|---:|---:|
| 20% | 13 | 0 | 8 | 0 | 9% | 0% |
| 40% | 10 | 0 | 6 | 0 | 9% | 0% |
| 60% | 8 | 0 | 5 | 0 | 9% | 0% |
| 80% | 5 | 0 | 3 | 0 | 9% | 0% |

A goal is paired when another player with minutes in that gameweek
has an assist. A run is a goal appearance with another goal by the
same player on a neighboring hidden fixture.

| revealed | actual goals with an assist beside them | painted |
|---|---|---|
| 20% | 703/730 (96%) | 0/0 |
| 40% | 525/544 (97%) | 0/0 |
| 60% | 346/360 (96%) | 0/0 |
| 80% | 191/199 (96%) | 0/0 |

| revealed | actual goals in a run | painted |
|---|---|---|
| 20% | 224/730 (31%) | 0/0 |
| 40% | 167/544 (31%) | 0/0 |
| 60% | 98/360 (27%) | 0/0 |
| 80% | 52/199 (26%) | 0/0 |

With the future event squares left gray, the share of appearances
that receive an event square. The squares are listed from the bottom.

| source | minutes | 1st square | 2nd square | 3rd square | any square |
|---|---|---:|---:|---:|---:|
| 20% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual goals | played | 0.081 | 0.008 | 0.001 | 0.081 |
| 20% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual assists | played | 0.078 | 0.005 | 0.000 | 0.078 |
| 20% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted goals | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted assists | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual goals | played | 0.079 | 0.007 | 0.001 | 0.079 |
| 40% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual assists | played | 0.077 | 0.004 | 0.000 | 0.077 |
| 40% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted goals | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted assists | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual goals | played | 0.080 | 0.007 | 0.001 | 0.080 |
| 60% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual assists | played | 0.078 | 0.004 | 0.000 | 0.078 |
| 60% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted goals | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted assists | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual goals | played | 0.082 | 0.008 | 0.000 | 0.082 |
| 80% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual assists | played | 0.079 | 0.005 | 0.000 | 0.079 |
| 80% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted goals | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted assists | played | 0.000 | 0.000 | 0.000 | 0.000 |

A painted stack skips a lower square when a higher square is filled
and a square below it is empty.

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 0 | 0 | 0 | 0 |
| 40% | 0 | 0 | 0 | 0 |
| 60% | 0 | 0 | 0 | 0 |
| 80% | 0 | 0 | 0 | 0 |
