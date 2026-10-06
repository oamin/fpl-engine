# Decision margin

The two readings locked in `reports/decision_margin_plan.md`, counted after the lock. The score is not changed. The pair's realised points are not on the stored signing row, so they do not enter this count and they do not choose the call.

## Reading A

The 64 signings the rebuild left out. The pair is that rebuild's lowest score at the position. The margin is that score minus the human's. Club or price is the constraint. The ranking bins are only the rows tagged neither.

Outside: 64 signings, -229. Held by the rebuild: 19 signings, -100. Constrained: 40 signings, -163, share 0.71. Unconstrained: 24 signings, -66.

The top-level call on the outside points is constraint. The share is already past a half.

| Margin | Signings | Points | Share of the unconstrained points |
|---|---:|---:|---:|
| <=0 | 0 | 0 | 0.00 |
| 0-0.5 | 12 | -31 | 0.47 |
| 0.5-1.25 | 8 | -29 | 0.44 |
| >1.25 | 4 | -6 | 0.09 |

The ranking call is inconclusive.

## Reading B

Seasons 2022/23 through 2025/26. Weeks 6 to 38. An eligible row has at least three prior appearances and expected minutes of at least 45. Rank is within position that week. An equal score keeps the earlier player id as the higher rank. Bias is mean score minus mean points. The top band is ranks 1 to 5.

The winner's-curse claim is parked. 2022/23 has no Gameweek 7 in the sheet, so the top band is 640 rows.

| Season | Eligible rows | Eligible bias | Ranks 1–5 bias | Gap |
|---|---:|---:|---:|---:|
| 2022/23 | 6909 | -0.25 | 0.11 | 0.35 |
| 2023/24 | 7156 | 0.06 | -0.04 | -0.10 |
| 2024/25 | 7237 | 0.11 | 0.17 | 0.06 |
| 2025/26 | 7286 | 0.07 | 0.23 | 0.16 |

Bands:

| Season | Band | Rows | Mean score | Mean points | Bias |
|---|---|---:|---:|---:|---:|
| 2022/23 | 1-5 | 640 | 4.65 | 4.54 | 0.11 |
| 2022/23 | 6-10 | 640 | 3.85 | 4.22 | -0.37 |
| 2022/23 | 11-20 | 1211 | 3.34 | 3.85 | -0.51 |
| 2022/23 | rest | 4418 | 2.62 | 2.83 | -0.21 |
| 2023/24 | 1-5 | 660 | 4.89 | 4.93 | -0.04 |
| 2023/24 | 6-10 | 657 | 4.05 | 4.29 | -0.24 |
| 2023/24 | 11-20 | 1258 | 3.53 | 3.58 | -0.05 |
| 2023/24 | rest | 4581 | 2.71 | 2.57 | 0.15 |
| 2024/25 | 1-5 | 660 | 4.90 | 4.73 | 0.17 |
| 2024/25 | 6-10 | 660 | 4.04 | 4.07 | -0.02 |
| 2024/25 | 11-20 | 1232 | 3.52 | 3.66 | -0.14 |
| 2024/25 | rest | 4685 | 2.72 | 2.53 | 0.19 |
| 2025/26 | 1-5 | 660 | 4.80 | 4.57 | 0.23 |
| 2025/26 | 6-10 | 660 | 4.09 | 4.27 | -0.17 |
| 2025/26 | 11-20 | 1260 | 3.65 | 3.63 | 0.02 |
| 2025/26 | rest | 4706 | 3.09 | 2.99 | 0.10 |

## Reading

The accessible disagreements do not gather in one bin. The score stays as it is. The winner's-curse claim is parked. Gemini kept the count ([decision margin](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)).
