# Stepped training and match odds

The gate paints the whole remainder at once and has no odds bar.
Stepped training shows the true past and asks only for the next gameweek,
with every later column removed. That week keeps minutes and the venue.
The odds bar is opening 1X2 prices after
the Shin de-vig: green home, grey draw, red away, from the top.
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
| 20% | stepped odds | goals | 730 | 672 | 181 | 224 | 370 |
| 20% | stepped odds | assists | 707 | 796 | 116 | 173 | 254 |
| 40% | stepped odds | goals | 544 | 516 | 138 | 170 | 285 |
| 40% | stepped odds | assists | 531 | 610 | 89 | 131 | 189 |
| 60% | stepped odds | goals | 360 | 340 | 89 | 111 | 184 |
| 60% | stepped odds | assists | 351 | 407 | 60 | 90 | 121 |
| 80% | stepped odds | goals | 199 | 181 | 52 | 63 | 94 |
| 80% | stepped odds | assists | 192 | 218 | 40 | 54 | 72 |
| 20% | painter odds | goals | 730 | 764 | 191 | 246 | 388 |
| 20% | painter odds | assists | 707 | 687 | 98 | 159 | 225 |
| 40% | painter odds | goals | 544 | 595 | 153 | 195 | 308 |
| 40% | painter odds | assists | 531 | 536 | 75 | 122 | 171 |
| 60% | painter odds | goals | 360 | 387 | 97 | 135 | 199 |
| 60% | painter odds | assists | 351 | 348 | 45 | 76 | 99 |
| 80% | painter odds | goals | 199 | 207 | 58 | 76 | 105 |
| 80% | painter odds | assists | 192 | 186 | 28 | 44 | 56 |

The gate stays the one-shot card without an odds bar: at 20% revealed,
192 goal bars on the same appearance and 257 actual goal bars within one gameweek.
The stepped rows are the 15:00 UK pass, with venue kept on the greyed week.
The painter rows are the 14:42 UK pass.

