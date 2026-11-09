# Header rows removed

The one-shot card keeps venue and drops concede, regulars, own history,
and opponent history. Those values are also removed from the numeric
match-up. The recent scoring rate stays. A second reader adds the odds
bar above the venue band, on every known week and every grey week.
The stepped reader uses the same card, asks for one gameweek, and
removes every later column.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | stripped | goals | 730 | 746 | 190 | 258 | 382 |
| 20% | stripped | assists | 707 | 681 | 103 | 183 | 221 |
| 40% | stripped | goals | 544 | 576 | 150 | 201 | 295 |
| 40% | stripped | assists | 531 | 523 | 74 | 134 | 156 |
| 60% | stripped | goals | 360 | 377 | 99 | 132 | 189 |
| 60% | stripped | assists | 351 | 341 | 45 | 87 | 94 |
| 80% | stripped | goals | 199 | 201 | 56 | 70 | 97 |
| 80% | stripped | assists | 192 | 182 | 24 | 49 | 52 |
| 20% | stripped odds | goals | 730 | 744 | 192 | 259 | 385 |
| 20% | stripped odds | assists | 707 | 679 | 101 | 179 | 222 |
| 40% | stripped odds | goals | 544 | 576 | 151 | 202 | 294 |
| 40% | stripped odds | assists | 531 | 522 | 74 | 138 | 157 |
| 60% | stripped odds | goals | 360 | 377 | 100 | 134 | 191 |
| 60% | stripped odds | assists | 351 | 340 | 42 | 88 | 90 |
| 80% | stripped odds | goals | 199 | 201 | 57 | 72 | 99 |
| 80% | stripped odds | assists | 192 | 183 | 23 | 49 | 52 |
| 20% | stripped step | goals | 730 | 624 | 168 | 217 | 343 |
| 20% | stripped step | assists | 707 | 674 | 89 | 172 | 207 |
| 40% | stripped step | goals | 544 | 478 | 128 | 166 | 259 |
| 40% | stripped step | assists | 531 | 517 | 66 | 127 | 151 |
| 60% | stripped step | goals | 360 | 312 | 79 | 106 | 162 |
| 60% | stripped step | assists | 351 | 336 | 39 | 82 | 95 |
| 80% | stripped step | goals | 199 | 165 | 47 | 60 | 86 |
| 80% | stripped step | assists | 192 | 178 | 24 | 49 | 54 |

The no-odds card is the gate. At 20% revealed it puts 190 goal bars
on the same appearance and 258 within one gameweek. The stepped reader,
scored at 16:06 UK, puts 168 and 217 there.

