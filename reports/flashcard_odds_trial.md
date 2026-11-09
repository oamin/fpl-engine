# Stepped training and match odds

The gate paints the whole remainder at once and has no odds bar.
Stepped training shows the true past and asks only for the next gameweek,
with every later column removed. The odds bar is opening 1X2 prices after
the Shin de-vig: green home, grey draw, red away, from the top.
The stepped card shows that bar on known weeks and the step week.
The full painter shows it on every known week and every grey week.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | stepped | goals | 730 | 699 | 180 | 228 | 373 |
| 20% | stepped | assists | 707 | 845 | 125 | 185 | 279 |
| 40% | stepped | goals | 544 | 536 | 136 | 171 | 286 |
| 40% | stepped | assists | 531 | 645 | 93 | 142 | 209 |
| 60% | stepped | goals | 360 | 351 | 85 | 108 | 182 |
| 60% | stepped | assists | 351 | 427 | 63 | 95 | 134 |
| 80% | stepped | goals | 199 | 189 | 50 | 61 | 93 |
| 80% | stepped | assists | 192 | 230 | 40 | 53 | 74 |
| 20% | stepped odds | goals | 730 | 675 | 180 | 224 | 371 |
| 20% | stepped odds | assists | 707 | 828 | 120 | 178 | 263 |
| 40% | stepped odds | goals | 544 | 519 | 136 | 169 | 284 |
| 40% | stepped odds | assists | 531 | 635 | 92 | 137 | 199 |
| 60% | stepped odds | goals | 360 | 340 | 87 | 110 | 183 |
| 60% | stepped odds | assists | 351 | 419 | 62 | 92 | 126 |
| 80% | stepped odds | goals | 199 | 183 | 50 | 62 | 93 |
| 80% | stepped odds | assists | 192 | 223 | 40 | 53 | 72 |
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
Wall clock: 172 seconds on CPU.

