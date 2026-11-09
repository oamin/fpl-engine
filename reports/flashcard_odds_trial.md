# Stepped training and match odds

The gate paints the whole remainder at once and has no odds bar.
Stepped training shows the true past and asks only for the next gameweek,
with every later column removed. That week keeps minutes and the venue.
The odds bar is opening 1X2 prices after
the Shin de-vig: blue home, yellow draw, purple away, from the top.
The stepped card shows that bar on known weeks and the step week.
The full painter shows it on every known week and every grey week.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | stepped | goals | 730 | 692 | 183 | 233 | 371 |
| 20% | stepped | assists | 707 | 798 | 115 | 181 | 261 |
| 40% | stepped | goals | 544 | 528 | 141 | 179 | 288 |
| 40% | stepped | assists | 531 | 610 | 89 | 138 | 196 |
| 60% | stepped | goals | 360 | 348 | 90 | 115 | 183 |
| 60% | stepped | assists | 351 | 400 | 61 | 96 | 129 |
| 80% | stepped | goals | 199 | 186 | 54 | 66 | 96 |
| 80% | stepped | assists | 192 | 213 | 40 | 55 | 73 |
| 20% | stepped odds | goals | 730 | 677 | 184 | 231 | 373 |
| 20% | stepped odds | assists | 707 | 810 | 117 | 183 | 260 |
| 40% | stepped odds | goals | 544 | 521 | 141 | 179 | 288 |
| 40% | stepped odds | assists | 531 | 620 | 89 | 136 | 194 |
| 60% | stepped odds | goals | 360 | 344 | 92 | 117 | 186 |
| 60% | stepped odds | assists | 351 | 411 | 61 | 89 | 125 |
| 80% | stepped odds | goals | 199 | 184 | 54 | 67 | 97 |
| 80% | stepped odds | assists | 192 | 220 | 40 | 54 | 72 |
| 20% | painter odds | goals | 730 | 764 | 193 | 259 | 390 |
| 20% | painter odds | assists | 707 | 686 | 101 | 184 | 223 |
| 40% | painter odds | goals | 544 | 598 | 150 | 200 | 303 |
| 40% | painter odds | assists | 531 | 536 | 72 | 136 | 161 |
| 60% | painter odds | goals | 360 | 387 | 95 | 128 | 193 |
| 60% | painter odds | assists | 351 | 349 | 42 | 84 | 93 |
| 80% | painter odds | goals | 199 | 207 | 56 | 73 | 101 |
| 80% | painter odds | assists | 192 | 186 | 19 | 45 | 47 |

The gate is the stripped card, with no odds bar: at 20% revealed,
190 goal bars on the same appearance and 258 actual goal bars within one gameweek.
The stepped rows without odds are the 15:00 UK pass. The odds rows are the
15:29 UK pass, after the bar moved to blue, yellow, and purple.

