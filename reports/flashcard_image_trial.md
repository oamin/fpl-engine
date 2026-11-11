# Image cell painters

The training set is the rendered card. Each square is one cell.
A future goal or assist square is gray, and the painter fills it.
Odds, venue, the opponent mark, the minute stack, and the position
color stay on the picture, including on a future gameweek. Each
finished card is one draw at temperature 1. The other squad is not
an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | maskgit | goals | 730 | 771 | 76 | 172 | 163 |
| 20% | maskgit | assists | 707 | 730 | 60 | 160 | 158 |
| 40% | maskgit | goals | 544 | 612 | 56 | 124 | 117 |
| 40% | maskgit | assists | 531 | 622 | 48 | 121 | 123 |
| 60% | maskgit | goals | 360 | 437 | 34 | 83 | 83 |
| 60% | maskgit | assists | 351 | 421 | 35 | 93 | 93 |
| 80% | maskgit | goals | 199 | 247 | 22 | 51 | 51 |
| 80% | maskgit | assists | 192 | 249 | 19 | 51 | 49 |
| 20% | column | goals | 730 | 1052 | 104 | 242 | 238 |
| 20% | column | assists | 707 | 1039 | 100 | 213 | 204 |
| 40% | column | goals | 544 | 826 | 73 | 163 | 161 |
| 40% | column | assists | 531 | 832 | 55 | 145 | 148 |
| 60% | column | goals | 360 | 537 | 46 | 121 | 121 |
| 60% | column | assists | 351 | 510 | 38 | 109 | 108 |
| 80% | column | goals | 199 | 278 | 25 | 55 | 54 |
| 80% | column | assists | 192 | 290 | 29 | 57 | 60 |
| 20% | var | goals | 730 | 606 | 51 | 137 | 119 |
| 20% | var | assists | 707 | 598 | 32 | 100 | 96 |
| 40% | var | goals | 544 | 464 | 39 | 101 | 87 |
| 40% | var | assists | 531 | 449 | 22 | 73 | 66 |
| 60% | var | goals | 360 | 305 | 27 | 65 | 57 |
| 60% | var | assists | 351 | 296 | 16 | 51 | 46 |
| 80% | var | goals | 199 | 158 | 12 | 25 | 23 |
| 80% | var | assists | 192 | 162 | 7 | 22 | 19 |

MaskGIT fills the gray squares in eight passes, keeping the sure
ones each time. The column raster paints the next gameweek from
the picture on its left, then writes those squares before the next
week. A band is drawn from the bottom square upward, and the square
above is drawn after the square below it is on the card. VAR paints
scales of 5, 10, 20, and 40. A coarse block carries the bottom
square only, and the full scale paints the squares above it. The
stored gate is the 21:47 UK full painter.
These runs do not replace it.

## maskgit

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 771 | 76 | 192 | -116 | 172 | 256 | -84 |
| 20% | assists | 707 | 730 | 60 | 100 | -40 | 160 | 161 | -1 |
| 40% | goals | 544 | 612 | 56 | 147 | -91 | 124 | 200 | -76 |
| 40% | assists | 531 | 622 | 48 | 78 | -30 | 121 | 119 | +2 |
| 60% | goals | 360 | 437 | 34 | 94 | -60 | 83 | 134 | -51 |
| 60% | assists | 351 | 421 | 35 | 55 | -20 | 93 | 84 | +9 |
| 80% | goals | 199 | 247 | 22 | 55 | -33 | 51 | 73 | -22 |
| 80% | assists | 192 | 249 | 19 | 36 | -17 | 51 | 51 | +0 |

## column

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 1052 | 104 | 192 | -88 | 242 | 256 | -14 |
| 20% | assists | 707 | 1039 | 100 | 100 | +0 | 213 | 161 | +52 |
| 40% | goals | 544 | 826 | 73 | 147 | -74 | 163 | 200 | -37 |
| 40% | assists | 531 | 832 | 55 | 78 | -23 | 145 | 119 | +26 |
| 60% | goals | 360 | 537 | 46 | 94 | -48 | 121 | 134 | -13 |
| 60% | assists | 351 | 510 | 38 | 55 | -17 | 109 | 84 | +25 |
| 80% | goals | 199 | 278 | 25 | 55 | -30 | 55 | 73 | -18 |
| 80% | assists | 192 | 290 | 29 | 36 | -7 | 57 | 51 | +6 |

## var

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 606 | 51 | 192 | -141 | 137 | 256 | -119 |
| 20% | assists | 707 | 598 | 32 | 100 | -68 | 100 | 161 | -61 |
| 40% | goals | 544 | 464 | 39 | 147 | -108 | 101 | 200 | -99 |
| 40% | assists | 531 | 449 | 22 | 78 | -56 | 73 | 119 | -46 |
| 60% | goals | 360 | 305 | 27 | 94 | -67 | 65 | 134 | -69 |
| 60% | assists | 351 | 296 | 16 | 55 | -39 | 51 | 84 | -33 |
| 80% | goals | 199 | 158 | 12 | 55 | -43 | 25 | 73 | -48 |
| 80% | assists | 192 | 162 | 7 | 36 | -29 | 22 | 51 | -29 |

A square keeps its row in the column. The bottom square of a band
sees a different window from the square above it, and the minute
stack is in that window. With the future event squares left gray,
the chance of painting each goal square, from the bottom, before
the draw:

| model | minutes | 1st square | 2nd square | 3rd square | any square |
|---|---|---:|---:|---:|---:|
| maskgit | zero | 0.002 | 0.001 | 0.001 | 0.004 |
| maskgit | played | 0.077 | 0.018 | 0.006 | 0.098 |
| column | zero | 0.003 | 0.001 | 0.001 | 0.004 |
| column | played | 0.085 | 0.019 | 0.006 | 0.107 |
| var | zero | 0.003 | 0.001 | 0.001 | 0.004 |
| var | played | 0.089 | 0.012 | 0.007 | 0.105 |

A painted stack skips a lower square when a higher square is filled
and a square below it is empty.

### maskgit

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 771 | 47 | 730 | 54 |
| 40% | 612 | 40 | 622 | 41 |
| 60% | 437 | 28 | 421 | 23 |
| 80% | 247 | 12 | 249 | 16 |

### column

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 1052 | 41 | 1039 | 41 |
| 40% | 826 | 37 | 832 | 35 |
| 60% | 537 | 26 | 510 | 27 |
| 80% | 278 | 10 | 290 | 13 |

### var

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 606 | 62 | 598 | 63 |
| 40% | 464 | 48 | 449 | 47 |
| 60% | 305 | 31 | 296 | 36 |
| 80% | 158 | 12 | 162 | 19 |
