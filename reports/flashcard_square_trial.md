# Square card

The odds row is sixteen squares. The venue mark is one square. The
opponent is one square, a club symbol. Minutes are six squares of fifteen
minutes, filled from the bottom. One goal or assist is one square, and
those rows are three squares tall. The position color is one square. The
full painter shows this card across the remainder. The stepped painter
keeps the odds, the venue, the opponent, and the minutes on the next
week and removes every later column. The other squad is not an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | square full | goals | 730 | 0 | 0 | 0 | 0 |
| 20% | square full | assists | 707 | 1 | 0 | 1 | 1 |
| 40% | square full | goals | 544 | 2 | 2 | 4 | 2 |
| 40% | square full | assists | 531 | 1 | 0 | 1 | 1 |
| 60% | square full | goals | 360 | 1 | 1 | 2 | 1 |
| 60% | square full | assists | 351 | 0 | 0 | 0 | 0 |
| 80% | square full | goals | 199 | 1 | 1 | 2 | 1 |
| 80% | square full | assists | 192 | 0 | 0 | 0 | 0 |
| 20% | square step | goals | 730 | 1 | 0 | 0 | 0 |
| 20% | square step | assists | 707 | 0 | 0 | 0 | 0 |
| 40% | square step | goals | 544 | 1 | 0 | 0 | 0 |
| 40% | square step | assists | 531 | 0 | 0 | 0 | 0 |
| 60% | square step | goals | 360 | 1 | 0 | 0 | 0 |
| 60% | square step | assists | 351 | 0 | 0 | 0 | 0 |
| 80% | square step | goals | 199 | 1 | 0 | 0 | 0 |
| 80% | square step | assists | 192 | 0 | 0 | 0 | 0 |

Each row paints the count with the highest probability. The class
for zero squares wins on almost every row, so the row stays empty.
A row holds three squares, and that is the only cap. There is no
fixture budget and no minutes gate. The stored gate is the 21:47 UK
full painter, which still used that budget. This run does not
replace it.

| revealed | channel | actual bars | same appearance | gate | difference | within 1 GW | gate | difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 0 | 192 | -192 | 0 | 256 | -256 |
| 20% | assists | 707 | 0 | 100 | -100 | 1 | 161 | -160 |
| 40% | goals | 544 | 2 | 147 | -145 | 4 | 200 | -196 |
| 40% | assists | 531 | 0 | 78 | -78 | 1 | 119 | -118 |
| 60% | goals | 360 | 1 | 94 | -93 | 2 | 134 | -132 |
| 60% | assists | 351 | 0 | 55 | -55 | 0 | 84 | -84 |
| 80% | goals | 199 | 1 | 55 | -54 | 2 | 73 | -71 |
| 80% | assists | 192 | 0 | 36 | -36 | 0 | 51 | -51 |

Wall clock: 157 seconds on CPU.

