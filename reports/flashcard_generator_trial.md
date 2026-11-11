# Card generators

Three painters train on the squares of the card. The known prefix
keeps its goal and assist squares. Odds, venue, opponent, minutes,
and position stay visible, including on a future gameweek. Each
finished card is one draw at temperature 1. The other squad is not
an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | maskgit | goals | 730 | 233 | 28 | 84 | 69 |
| 20% | maskgit | assists | 707 | 149 | 14 | 36 | 30 |
| 40% | maskgit | goals | 544 | 185 | 28 | 64 | 52 |
| 40% | maskgit | assists | 531 | 121 | 9 | 30 | 25 |
| 60% | maskgit | goals | 360 | 114 | 15 | 35 | 33 |
| 60% | maskgit | assists | 351 | 84 | 10 | 27 | 23 |
| 80% | maskgit | goals | 199 | 60 | 8 | 19 | 18 |
| 80% | maskgit | assists | 192 | 44 | 3 | 6 | 4 |
| 20% | column | goals | 730 | 861 | 51 | 140 | 127 |
| 20% | column | assists | 707 | 784 | 35 | 96 | 91 |
| 40% | column | goals | 544 | 725 | 44 | 125 | 108 |
| 40% | column | assists | 531 | 632 | 27 | 82 | 77 |
| 60% | column | goals | 360 | 456 | 19 | 66 | 59 |
| 60% | column | assists | 351 | 435 | 17 | 45 | 43 |
| 80% | column | goals | 199 | 235 | 12 | 28 | 28 |
| 80% | column | assists | 192 | 201 | 6 | 22 | 22 |
| 20% | var | goals | 730 | 1098 | 101 | 211 | 214 |
| 20% | var | assists | 707 | 1080 | 61 | 143 | 148 |
| 40% | var | goals | 544 | 814 | 74 | 154 | 159 |
| 40% | var | assists | 531 | 836 | 39 | 105 | 116 |
| 60% | var | goals | 360 | 512 | 43 | 98 | 96 |
| 60% | var | assists | 351 | 504 | 36 | 86 | 87 |
| 80% | var | goals | 199 | 323 | 37 | 70 | 68 |
| 80% | var | assists | 192 | 293 | 15 | 47 | 47 |

MaskGIT fills the open squares in eight passes, keeping the sure
ones each time. The column raster paints the next gameweek from
the weeks on its left. VAR paints scales of 5, 10, 20, and 40,
and a coarse square is the highest count in its block. The stored
gate is the 21:47 UK full painter. These runs do not replace it.

## maskgit

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 233 | 28 | 192 | -164 | 84 | 256 | -172 |
| 20% | assists | 707 | 149 | 14 | 100 | -86 | 36 | 161 | -125 |
| 40% | goals | 544 | 185 | 28 | 147 | -119 | 64 | 200 | -136 |
| 40% | assists | 531 | 121 | 9 | 78 | -69 | 30 | 119 | -89 |
| 60% | goals | 360 | 114 | 15 | 94 | -79 | 35 | 134 | -99 |
| 60% | assists | 351 | 84 | 10 | 55 | -45 | 27 | 84 | -57 |
| 80% | goals | 199 | 60 | 8 | 55 | -47 | 19 | 73 | -54 |
| 80% | assists | 192 | 44 | 3 | 36 | -33 | 6 | 51 | -45 |

## column

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 861 | 51 | 192 | -141 | 140 | 256 | -116 |
| 20% | assists | 707 | 784 | 35 | 100 | -65 | 96 | 161 | -65 |
| 40% | goals | 544 | 725 | 44 | 147 | -103 | 125 | 200 | -75 |
| 40% | assists | 531 | 632 | 27 | 78 | -51 | 82 | 119 | -37 |
| 60% | goals | 360 | 456 | 19 | 94 | -75 | 66 | 134 | -68 |
| 60% | assists | 351 | 435 | 17 | 55 | -38 | 45 | 84 | -39 |
| 80% | goals | 199 | 235 | 12 | 55 | -43 | 28 | 73 | -45 |
| 80% | assists | 192 | 201 | 6 | 36 | -30 | 22 | 51 | -29 |

## var

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 1098 | 101 | 192 | -91 | 211 | 256 | -45 |
| 20% | assists | 707 | 1080 | 61 | 100 | -39 | 143 | 161 | -18 |
| 40% | goals | 544 | 814 | 74 | 147 | -73 | 154 | 200 | -46 |
| 40% | assists | 531 | 836 | 39 | 78 | -39 | 105 | 119 | -14 |
| 60% | goals | 360 | 512 | 43 | 94 | -51 | 98 | 134 | -36 |
| 60% | assists | 351 | 504 | 36 | 55 | -19 | 86 | 84 | +2 |
| 80% | goals | 199 | 323 | 37 | 55 | -18 | 70 | 73 | -3 |
| 80% | assists | 192 | 293 | 15 | 36 | -21 | 47 | 51 | -4 |

Wall clock: 52 seconds on CPU.

