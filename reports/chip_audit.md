# Chip audit

The twelve stored chip climbs were replayed. The chip rule, the margins of 12 and 16, and `data/processed/half_plan_scores.csv` were left as they are. A replay that changed a chip week, a transfer count, a hit count, or a season total would have stopped with no split.

The wildcard gap is the decision-time eleven, rebuilt minus held. Priced steps are the next three club weeks. The tail is every later week in the half, including weeks that repeat the last priced step. The Gameweek 4 count is the twelve early wildcards, one on each climb. Later wildcards are listed on their own.

The lift is the stored chip total minus the stored empty total. Active is the chip week only: bench points on Bench Boost, one extra captain copy on Triple Captain, and the played eleven minus the squad in hand on Wildcard and Free Hit. The path gap is the rest of those chip weeks against the empty climb. The residual is every week with no chip. The three pieces add to the lift. Hits the wildcard avoided are not simulated.

Churn starts the week after each wildcard and stops before the next one. No new transfer search was run.

Gemini kept this split on 2026-10-05 ([chip audit](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). Ten of the twelve Gameweek 4 wildcards still clear 16 on the priced steps. The chip week is positive on every climb. In 2022/23 that week is worth 27 to 53 points and the weeks with no chip lose 77 to 112, so the season loss is the squad after the wildcard. Active points are not the majority of the positive lifts on 2023/24 Premium, 2025/26 Premium, or 2025/26 Template. All eight second-half wildcards in 2023/24 and 2025/26 miss 16 on the priced steps, and the copied tail is what clears them. Bounding the wildcard sum to the priced steps stays a later batch. The margins stay 12 and 16, and the chip rule stays as it is.

Gameweek 4 wildcards that clear 16 on priced steps alone: 10 of 12.

## Gameweek 4

| Season | Squad | GW | Priced gap | Tail gap | Full gap | Tail share | Priced clears 16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-23 | Next | 4 | 23.5 | 87.8 | 111.3 | 0.79 | yes |
| 2022-23 | Premium | 4 | 19.8 | 59.8 | 79.7 | 0.75 | yes |
| 2022-23 | Template | 4 | 10.4 | 29.0 | 39.4 | 0.74 | no |
| 2022-23 | Third | 4 | 6.7 | 24.0 | 30.6 | 0.78 | no |
| 2023-24 | Next | 4 | 68.4 | 291.4 | 359.8 | 0.81 | yes |
| 2023-24 | Premium | 4 | 72.5 | 316.5 | 389.0 | 0.81 | yes |
| 2023-24 | Template | 4 | 47.5 | 157.1 | 204.6 | 0.77 | yes |
| 2023-24 | Third | 4 | 49.7 | 249.7 | 299.4 | 0.83 | yes |
| 2025-26 | Next | 4 | 39.6 | 99.5 | 139.0 | 0.72 | yes |
| 2025-26 | Premium | 4 | 73.7 | 252.7 | 326.4 | 0.77 | yes |
| 2025-26 | Template | 4 | 60.4 | 235.1 | 295.5 | 0.80 | yes |
| 2025-26 | Third | 4 | 86.7 | 346.4 | 433.1 | 0.80 | yes |

## Later wildcards

| Season | Squad | GW | Priced gap | Tail gap | Full gap | Tail share | Priced clears 16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-23 | Next | 21 | 38.3 | 143.5 | 181.8 | 0.79 | yes |
| 2022-23 | Premium | 21 | 38.5 | 145.6 | 184.0 | 0.79 | yes |
| 2022-23 | Template | 21 | 45.5 | 152.1 | 197.6 | 0.77 | yes |
| 2022-23 | Third | 21 | 38.3 | 143.5 | 181.8 | 0.79 | yes |
| 2023-24 | Next | 20 | 5.7 | 59.9 | 65.6 | 0.91 | no |
| 2023-24 | Premium | 20 | 12.2 | 94.1 | 106.4 | 0.89 | no |
| 2023-24 | Template | 20 | 14.1 | 130.1 | 144.3 | 0.90 | no |
| 2023-24 | Third | 20 | 9.5 | 87.4 | 96.8 | 0.90 | no |
| 2025-26 | Next | 24 | 2.4 | 23.7 | 26.1 | 0.91 | no |
| 2025-26 | Premium | 24 | 2.2 | 23.1 | 25.3 | 0.91 | no |
| 2025-26 | Template | 29 | 8.2 | 46.8 | 55.1 | 0.85 | no |
| 2025-26 | Third | 24 | 4.3 | 19.0 | 23.3 | 0.82 | no |

## Lift

| Season | Squad | Lift | Active | Path gap | Residual |
| --- | --- | --- | --- | --- | --- |
| 2022-23 | Next | -57 | 28.0 | -2.0 | -83.0 |
| 2022-23 | Premium | -70 | 41.0 | -26.0 | -85.0 |
| 2022-23 | Template | -96 | 53.0 | -37.0 | -112.0 |
| 2022-23 | Third | -40 | 27.0 | 10.0 | -77.0 |
| 2023-24 | Next | -216 | 36.0 | -37.0 | -215.0 |
| 2023-24 | Premium | 100 | 46.0 | -16.0 | 70.0 |
| 2023-24 | Template | -15 | 49.0 | -14.0 | -50.0 |
| 2023-24 | Third | 128 | 86.0 | 2.0 | 40.0 |
| 2025-26 | Next | 34 | 74.0 | -2.0 | -38.0 |
| 2025-26 | Premium | 130 | 50.0 | -23.0 | 103.0 |
| 2025-26 | Template | 199 | 70.0 | 77.0 | 52.0 |
| 2025-26 | Third | 177 | 93.0 | 41.0 | 43.0 |

## Churn after the wildcard

| Season | Squad | Wildcard | Until | Chip transfers | Empty transfers | Chip hits | Empty hits | Chip bank | Empty bank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-23 | Next | 4 | 21 | 20 | 18 | 3 | 3 | 5 | 22 |
| 2022-23 | Next | 21 |  | 20 | 21 | 4 | 4 | 1 | 0 |
| 2022-23 | Premium | 4 | 21 | 20 | 20 | 3 | 4 | 10 | 38 |
| 2022-23 | Premium | 21 |  | 20 | 22 | 4 | 5 | 0 | 10 |
| 2022-23 | Template | 4 | 21 | 20 | 20 | 3 | 4 | 51 | 2 |
| 2022-23 | Template | 21 |  | 23 | 21 | 6 | 5 | 9 | 2 |
| 2022-23 | Third | 4 | 21 | 20 | 19 | 3 | 2 | 7 | 67 |
| 2022-23 | Third | 21 |  | 21 | 20 | 4 | 4 | 6 | 4 |
| 2023-24 | Next | 4 | 20 | 16 | 15 | 0 | 1 | 1 | 11 |
| 2023-24 | Next | 20 |  | 19 | 19 | 2 | 3 | 1 | 2 |
| 2023-24 | Premium | 4 | 20 | 16 | 16 | 0 | 2 | 3 | 14 |
| 2023-24 | Premium | 20 |  | 19 | 18 | 2 | 3 | 4 | 0 |
| 2023-24 | Template | 4 | 20 | 16 | 15 | 0 | 0 | 11 | 8 |
| 2023-24 | Template | 20 |  | 19 | 18 | 2 | 2 | 3 | 0 |
| 2023-24 | Third | 4 | 20 | 12 | 15 | 0 | 1 | 6 | 6 |
| 2023-24 | Third | 20 |  | 33 | 19 | 3 | 2 | 1 | 0 |
| 2025-26 | Next | 4 | 24 | 20 | 20 | 0 | 2 | 2 | 4 |
| 2025-26 | Next | 24 |  | 8 | 12 | 0 | 0 | 12 | 5 |
| 2025-26 | Premium | 4 | 24 | 18 | 18 | 0 | 1 | 1 | 18 |
| 2025-26 | Premium | 24 |  | 9 | 12 | 0 | 0 | 1 | 5 |
| 2025-26 | Template | 4 | 29 | 19 | 23 | 0 | 3 | 3 | 0 |
| 2025-26 | Template | 29 |  | 10 | 9 | 0 | 0 | 11 | 2 |
| 2025-26 | Third | 4 | 24 | 19 | 19 | 0 | 2 | 10 | 14 |
| 2025-26 | Third | 24 |  | 13 | 14 | 0 | 0 | 7 | 14 |

The quote beside the stored chip file stays Next +34 (2171 to 2205) and Premium 2214.

Wildcard rows are in `data/processed/chip_audit_wildcards.csv`. Lift rows are in `data/processed/chip_audit_lifts.csv`. Churn rows are in `data/processed/chip_audit_churn.csv`.
