# Remainder painter

The training set is the rendered card. Each square is one cell.
A future goal or assist square is gray. The painter reads the whole
card once, draws one code for the gray remainder, and paints every
gray square from that code and the visible card. Odds, venue, the
opponent mark, the minute stack, and the position color stay on the
picture, including on a future gameweek. The code is one of the
remainder patterns learned from the training pictures. Each finished
card is one draw at temperature 1. The other squad is not an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 1201 | 274 | 346 | 557 |
| 20% | assists | 707 | 1125 | 193 | 271 | 385 |
| 40% | goals | 544 | 913 | 207 | 265 | 426 |
| 40% | assists | 531 | 793 | 130 | 189 | 254 |
| 60% | goals | 360 | 612 | 136 | 172 | 277 |
| 60% | assists | 351 | 528 | 91 | 132 | 171 |
| 80% | goals | 199 | 339 | 79 | 103 | 153 |
| 80% | assists | 192 | 274 | 49 | 68 | 84 |

One code is drawn for the whole remainder. The paint of every future
event square is fixed by that code and the visible card. The stored
gate is the 21:47 UK full painter. This run does not replace it.

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 1201 | 274 | 192 | +82 | 346 | 256 | +90 |
| 20% | assists | 707 | 1125 | 193 | 100 | +93 | 271 | 161 | +110 |
| 40% | goals | 544 | 913 | 207 | 147 | +60 | 265 | 200 | +65 |
| 40% | assists | 531 | 793 | 130 | 78 | +52 | 189 | 119 | +70 |
| 60% | goals | 360 | 612 | 136 | 94 | +42 | 172 | 134 | +38 |
| 60% | assists | 351 | 528 | 91 | 55 | +36 | 132 | 84 | +48 |
| 80% | goals | 199 | 339 | 79 | 55 | +24 | 103 | 73 | +30 |
| 80% | assists | 192 | 274 | 49 | 36 | +13 | 68 | 51 | +17 |

A goal is paired when another player with minutes in that gameweek
has an assist. A run is a goal appearance with another goal by the
same player on a neighboring hidden fixture.

| revealed | actual goals with an assist beside them | painted |
|---|---|---|
| 20% | 703/730 (96%) | 935/1201 (78%) |
| 40% | 525/544 (97%) | 693/913 (76%) |
| 60% | 346/360 (96%) | 467/612 (76%) |
| 80% | 191/199 (96%) | 245/339 (72%) |

| revealed | actual goals in a run | painted |
|---|---|---|
| 20% | 224/730 (31%) | 914/1201 (76%) |
| 40% | 167/544 (31%) | 674/913 (74%) |
| 60% | 98/360 (27%) | 439/612 (72%) |
| 80% | 52/199 (26%) | 225/339 (66%) |

With the future event squares left gray, the share of appearances
that receive an event square. The squares are listed from the bottom.

| source | minutes | 1st square | 2nd square | 3rd square | any square |
|---|---|---:|---:|---:|---:|
| 20% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual goals | played | 0.081 | 0.008 | 0.001 | 0.081 |
| 20% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual assists | played | 0.078 | 0.005 | 0.000 | 0.078 |
| 20% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted goals | played | 0.133 | 0.001 | 0.000 | 0.133 |
| 20% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted assists | played | 0.124 | 0.000 | 0.000 | 0.124 |
| 40% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual goals | played | 0.079 | 0.007 | 0.001 | 0.079 |
| 40% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual assists | played | 0.077 | 0.004 | 0.000 | 0.077 |
| 40% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted goals | played | 0.132 | 0.001 | 0.000 | 0.132 |
| 40% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted assists | played | 0.115 | 0.000 | 0.000 | 0.115 |
| 60% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual goals | played | 0.080 | 0.007 | 0.001 | 0.080 |
| 60% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual assists | played | 0.078 | 0.004 | 0.000 | 0.078 |
| 60% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted goals | played | 0.135 | 0.001 | 0.000 | 0.135 |
| 60% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted assists | played | 0.117 | 0.000 | 0.000 | 0.117 |
| 80% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual goals | played | 0.082 | 0.008 | 0.000 | 0.082 |
| 80% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual assists | played | 0.079 | 0.005 | 0.000 | 0.079 |
| 80% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted goals | played | 0.140 | 0.001 | 0.000 | 0.140 |
| 80% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted assists | played | 0.113 | 0.000 | 0.000 | 0.113 |

A painted stack skips a lower square when a higher square is filled
and a square below it is empty.

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 1201 | 0 | 1125 | 0 |
| 40% | 913 | 0 | 793 | 0 |
| 60% | 612 | 0 | 528 | 0 |
| 80% | 339 | 0 | 274 | 0 |
