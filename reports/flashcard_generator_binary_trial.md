# Binary mask generators

The same three painters, with a binary mask. Each event row is
three squares, and each square is on or off. A hidden square is
off, and its mask bit is on. A draw turns those bits into a count
from zero to three. Odds, venue, opponent, minutes, and position
stay visible. The other squad is not an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | maskgit | goals | 730 | 333 | 24 | 64 | 49 |
| 20% | maskgit | assists | 707 | 302 | 11 | 39 | 36 |
| 40% | maskgit | goals | 544 | 255 | 9 | 39 | 34 |
| 40% | maskgit | assists | 531 | 211 | 7 | 24 | 22 |
| 60% | maskgit | goals | 360 | 175 | 12 | 31 | 28 |
| 60% | maskgit | assists | 351 | 152 | 8 | 16 | 16 |
| 80% | maskgit | goals | 199 | 100 | 6 | 15 | 13 |
| 80% | maskgit | assists | 192 | 85 | 3 | 9 | 9 |
| 20% | column | goals | 730 | 1599 | 72 | 190 | 181 |
| 20% | column | assists | 707 | 1390 | 67 | 184 | 171 |
| 40% | column | goals | 544 | 1296 | 66 | 149 | 150 |
| 40% | column | assists | 531 | 1099 | 48 | 145 | 146 |
| 60% | column | goals | 360 | 795 | 40 | 97 | 91 |
| 60% | column | assists | 351 | 703 | 34 | 91 | 87 |
| 80% | column | goals | 199 | 409 | 22 | 52 | 48 |
| 80% | column | assists | 192 | 368 | 25 | 55 | 50 |
| 20% | var | goals | 730 | 2233 | 126 | 312 | 312 |
| 20% | var | assists | 707 | 2035 | 102 | 263 | 269 |
| 40% | var | goals | 544 | 1625 | 97 | 213 | 217 |
| 40% | var | assists | 531 | 1529 | 80 | 190 | 210 |
| 60% | var | goals | 360 | 1048 | 65 | 154 | 162 |
| 60% | var | assists | 351 | 988 | 54 | 140 | 146 |
| 80% | var | goals | 199 | 588 | 41 | 85 | 91 |
| 80% | var | assists | 192 | 504 | 25 | 65 | 73 |

MaskGIT fills the open squares in eight passes, keeping the sure
ones each time. The column raster paints the next gameweek from
the weeks on its left. VAR paints scales of 5, 10, 20, and 40.
On the binary mask a coarse square is the highest count in its
block, written as on/off squares.
The stored gate is the 21:47 UK full painter. These runs do not
replace it.

## maskgit

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 333 | 24 | 192 | -168 | 64 | 256 | -192 |
| 20% | assists | 707 | 302 | 11 | 100 | -89 | 39 | 161 | -122 |
| 40% | goals | 544 | 255 | 9 | 147 | -138 | 39 | 200 | -161 |
| 40% | assists | 531 | 211 | 7 | 78 | -71 | 24 | 119 | -95 |
| 60% | goals | 360 | 175 | 12 | 94 | -82 | 31 | 134 | -103 |
| 60% | assists | 351 | 152 | 8 | 55 | -47 | 16 | 84 | -68 |
| 80% | goals | 199 | 100 | 6 | 55 | -49 | 15 | 73 | -58 |
| 80% | assists | 192 | 85 | 3 | 36 | -33 | 9 | 51 | -42 |

## column

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 1599 | 72 | 192 | -120 | 190 | 256 | -66 |
| 20% | assists | 707 | 1390 | 67 | 100 | -33 | 184 | 161 | +23 |
| 40% | goals | 544 | 1296 | 66 | 147 | -81 | 149 | 200 | -51 |
| 40% | assists | 531 | 1099 | 48 | 78 | -30 | 145 | 119 | +26 |
| 60% | goals | 360 | 795 | 40 | 94 | -54 | 97 | 134 | -37 |
| 60% | assists | 351 | 703 | 34 | 55 | -21 | 91 | 84 | +7 |
| 80% | goals | 199 | 409 | 22 | 55 | -33 | 52 | 73 | -21 |
| 80% | assists | 192 | 368 | 25 | 36 | -11 | 55 | 51 | +4 |

## var

| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 2233 | 126 | 192 | -66 | 312 | 256 | +56 |
| 20% | assists | 707 | 2035 | 102 | 100 | +2 | 263 | 161 | +102 |
| 40% | goals | 544 | 1625 | 97 | 147 | -50 | 213 | 200 | +13 |
| 40% | assists | 531 | 1529 | 80 | 78 | +2 | 190 | 119 | +71 |
| 60% | goals | 360 | 1048 | 65 | 94 | -29 | 154 | 134 | +20 |
| 60% | assists | 351 | 988 | 54 | 55 | -1 | 140 | 84 | +56 |
| 80% | goals | 199 | 588 | 41 | 55 | -14 | 85 | 73 | +12 |
| 80% | assists | 192 | 504 | 25 | 36 | -11 | 65 | 51 | +14 |

Wall clock: 52 seconds on CPU.

