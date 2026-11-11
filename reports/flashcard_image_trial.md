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
| 20% | maskgit | goals | 730 | 966 | 55 | 146 | 138 |
| 20% | maskgit | assists | 707 | 1000 | 49 | 127 | 133 |
| 40% | maskgit | goals | 544 | 708 | 36 | 96 | 91 |
| 40% | maskgit | assists | 531 | 779 | 33 | 92 | 98 |
| 60% | maskgit | goals | 360 | 460 | 26 | 68 | 66 |
| 60% | maskgit | assists | 351 | 537 | 17 | 61 | 64 |
| 80% | maskgit | goals | 199 | 264 | 18 | 45 | 43 |
| 80% | maskgit | assists | 192 | 293 | 18 | 34 | 35 |
| 20% | column | goals | 730 | 873 | 47 | 125 | 112 |
| 20% | column | assists | 707 | 887 | 47 | 117 | 112 |
| 40% | column | goals | 544 | 693 | 48 | 105 | 97 |
| 40% | column | assists | 531 | 670 | 35 | 98 | 88 |
| 60% | column | goals | 360 | 473 | 23 | 66 | 62 |
| 60% | column | assists | 351 | 473 | 22 | 76 | 76 |
| 80% | column | goals | 199 | 250 | 29 | 46 | 45 |
| 80% | column | assists | 192 | 248 | 17 | 36 | 35 |
| 20% | var | goals | 730 | 2294 | 107 | 255 | 246 |
| 20% | var | assists | 707 | 1652 | 67 | 185 | 188 |
| 40% | var | goals | 544 | 1801 | 86 | 198 | 193 |
| 40% | var | assists | 531 | 1279 | 47 | 123 | 129 |
| 60% | var | goals | 360 | 1212 | 53 | 120 | 119 |
| 60% | var | assists | 351 | 873 | 30 | 87 | 92 |
| 80% | var | goals | 199 | 680 | 28 | 61 | 64 |
| 80% | var | assists | 192 | 446 | 12 | 39 | 43 |

MaskGIT fills the gray squares in eight passes, keeping the sure
ones each time. The column raster paints the next gameweek from
the picture on its left, then writes those squares before the next
week. VAR paints scales of 5, 10, 20, and 40. A coarse square is
filled when any square in its block is filled, and the next scale
repaints the block. The stored gate is the 21:47 UK full painter.
These runs do not replace it.

## maskgit

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 966 | 55 | 192 | -137 | 146 | 256 | -110 |
| 20% | assists | 707 | 1000 | 49 | 100 | -51 | 127 | 161 | -34 |
| 40% | goals | 544 | 708 | 36 | 147 | -111 | 96 | 200 | -104 |
| 40% | assists | 531 | 779 | 33 | 78 | -45 | 92 | 119 | -27 |
| 60% | goals | 360 | 460 | 26 | 94 | -68 | 68 | 134 | -66 |
| 60% | assists | 351 | 537 | 17 | 55 | -38 | 61 | 84 | -23 |
| 80% | goals | 199 | 264 | 18 | 55 | -37 | 45 | 73 | -28 |
| 80% | assists | 192 | 293 | 18 | 36 | -18 | 34 | 51 | -17 |

## column

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 873 | 47 | 192 | -145 | 125 | 256 | -131 |
| 20% | assists | 707 | 887 | 47 | 100 | -53 | 117 | 161 | -44 |
| 40% | goals | 544 | 693 | 48 | 147 | -99 | 105 | 200 | -95 |
| 40% | assists | 531 | 670 | 35 | 78 | -43 | 98 | 119 | -21 |
| 60% | goals | 360 | 473 | 23 | 94 | -71 | 66 | 134 | -68 |
| 60% | assists | 351 | 473 | 22 | 55 | -33 | 76 | 84 | -8 |
| 80% | goals | 199 | 250 | 29 | 55 | -26 | 46 | 73 | -27 |
| 80% | assists | 192 | 248 | 17 | 36 | -19 | 36 | 51 | -15 |

## var

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 2294 | 107 | 192 | -85 | 255 | 256 | -1 |
| 20% | assists | 707 | 1652 | 67 | 100 | -33 | 185 | 161 | +24 |
| 40% | goals | 544 | 1801 | 86 | 147 | -61 | 198 | 200 | -2 |
| 40% | assists | 531 | 1279 | 47 | 78 | -31 | 123 | 119 | +4 |
| 60% | goals | 360 | 1212 | 53 | 94 | -41 | 120 | 134 | -14 |
| 60% | assists | 351 | 873 | 30 | 55 | -25 | 87 | 84 | +3 |
| 80% | goals | 199 | 680 | 28 | 55 | -27 | 61 | 73 | -12 |
| 80% | assists | 192 | 446 | 12 | 36 | -24 | 39 | 51 | -12 |

With the future event squares left gray, the chance that an appearance
gets any goal square, before the draw:

| model | zero minutes | minutes played |
|---|---:|---:|
| maskgit | 0.043 | 0.063 |
| column | 0.045 | 0.067 |
| var | 0.279 | 0.300 |
