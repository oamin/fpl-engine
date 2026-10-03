# Stage 45 — score repairs, fast XI

Seven scores, locked before these totals. Gameweeks 5–38. The fast XI does not carry a squad. A score 100 or more behind `score_xp` on any season is killed. Ahead on every season is the only route to a 2025/26 free-transfer climb, and the pass bar there is +34. `score_xp` stays the published score.

The 0.25 on defenders and forwards was read off the 2025/26 calibration. That season does not confirm `pos_shift`.

| arm | 2022/23 | 2023/24 | 2024/25 | 2025/26 | worst | result |
|---|---:|---:|---:|---:|---:|---|
| xp | 1803 | 2188 | 2150 | 2083 | — | baseline |
| no_defcon | +0 | +0 | +0 | -85 | -85 | alive |
| no_cs | -159 | -137 | -15 | -92 | -159 | killed |
| cs_half | +3 | -67 | +40 | +14 | -67 | alive |
| appear_linear | -29 | +41 | +12 | +26 | -29 | alive |
| gk6 | +0 | +0 | -3 | +2 | -3 | alive |
| gc_half | +33 | -24 | +19 | -54 | -54 | alive |
| pos_shift | -14 | -35 | +24 | -3 | -35 | alive |

Deltas are captained XI points minus the published fast XI.

## Defensive contributions

Mean `xp_defcon` on the buy pool. A positive number before 2025/26 is the model charging an award the official points did not pay.

| season | position | n | mean |
|---|---|---:|---:|
| 2022-23 | DEF | 2791 | 0.000 |
| 2022-23 | FWD | 765 | 0.000 |
| 2022-23 | GKP | 641 | 0.000 |
| 2022-23 | MID | 3310 | 0.000 |
| 2022-23 | ALL | 7507 | 0.000 |
| 2023-24 | DEF | 2860 | 0.000 |
| 2023-24 | FWD | 812 | 0.000 |
| 2023-24 | GKP | 647 | 0.000 |
| 2023-24 | MID | 3243 | 0.000 |
| 2023-24 | ALL | 7562 | 0.000 |
| 2024-25 | DEF | 2746 | 0.000 |
| 2024-25 | FWD | 688 | 0.000 |
| 2024-25 | GKP | 634 | 0.000 |
| 2024-25 | MID | 3464 | 0.000 |
| 2024-25 | ALL | 7532 | 0.000 |
| 2025-26 | DEF | 2899 | 0.453 |
| 2025-26 | FWD | 766 | 0.010 |
| 2025-26 | GKP | 639 | 0.000 |
| 2025-26 | MID | 3265 | 0.254 |
| 2025-26 | ALL | 7569 | 0.284 |

## Captain extra, published XI

| season | score_xp | goals | oracle | goals minus score | oracle minus score |
|---|---:|---:|---:|---:|---:|
| 2022-23 | 140 | 168 | 394 | +28 | +254 |
| 2023-24 | 216 | 211 | 481 | -5 | +265 |
| 2024-25 | 213 | 216 | 454 | +3 | +241 |
| 2025-26 | 230 | 210 | 430 | -20 | +200 |

Goalkeeper goals rescaled to 6 move the fast XI by 3 points in 2024/25 and by 2 in 2025/26. The formula leaves every outfield row equal to `score_xp`. Those points are a different eleven. Gemini cleared that as contamination and parked the arm: the four-season gap is about a point.

The fast XI is built from rows with minutes above 0. A player who did not play has no row, so these gaps reorder players who played.

Gemini parked every arm. `appear_linear` is the best of the batch: +41, +12, and +26, and −29 in 2022/23, about +50 across the four screens. It is not ahead on every season, so it does not take a free-transfer climb. `no_cs` is killed. Nothing replaces `score_xp`.

## Published shapes

| season | shape | weeks | points |
|---|---|---:|---:|
| 2022-23 | 1-3-4-3 | 20 | 1131 |
| 2022-23 | 1-3-5-2 | 2 | 129 |
| 2022-23 | 1-5-4-1 | 11 | 543 |
| 2023-24 | 1-3-4-3 | 12 | 756 |
| 2023-24 | 1-3-5-2 | 21 | 1373 |
| 2023-24 | 1-4-5-1 | 1 | 59 |
| 2024-25 | 1-3-4-3 | 15 | 867 |
| 2024-25 | 1-3-5-2 | 19 | 1283 |
| 2025-26 | 1-3-4-3 | 2 | 138 |
| 2025-26 | 1-3-5-2 | 5 | 260 |
| 2025-26 | 1-4-3-3 | 3 | 211 |
| 2025-26 | 1-4-4-2 | 5 | 322 |
| 2025-26 | 1-4-5-1 | 14 | 791 |
| 2025-26 | 1-5-3-2 | 1 | 79 |
| 2025-26 | 1-5-4-1 | 4 | 282 |
