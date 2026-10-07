# Stage 47 — crowd context

Deadline transfer flow and ownership are checked as their own columns. `score_xp` is the only score. Nothing in this file is added to it, and no eleven is picked.

The flow is `log(1 + transfers in) − log(1 + transfers out)`. That is the difference of the two log flows. Ownership is the share of managers who own the player, entered in units of 10 percentage points and centered on the training-fold mean. Volume is a within-week standard deviation on the eligible rows. The fit leaves out one season at a time, using 2022/23, 2023/24, 2024/25, and 2025/26. The holdout error is mean absolute error on Gameweeks 5–38, among players with at least 45 expected minutes and at least three prior appearances. A zero-minute week is absent, because the player log drops it.

The crowd columns carry held-out information only when the error falls in every left-out season and the mean fall is at least 0.02 points per player-week. The interaction is held to that same bar against the additive fit. Either result stays out of the transfer search.

## Holdout error

| holdout | rows | baseline MAE | additive MAE | flow and ownership | interaction MAE | interaction |
|---|---:|---:|---:|---:|---:|---:|
| 2022-23 | 7085 | 2.2184 | 2.2064 | +0.0121 | 2.2067 | -0.0003 |
| 2023-24 | 7330 | 2.2663 | 2.2541 | +0.0122 | 2.2551 | -0.0009 |
| 2024-25 | 7427 | 2.1964 | 2.1930 | +0.0035 | 2.1932 | -0.0002 |
| 2025-26 | 7464 | 2.2702 | 2.2553 | +0.0149 | 2.2554 | -0.0001 |

Mean error change, flow and ownership: **+0.0106**. The bar misses.

Mean error change, interaction against the additive fit: **-0.0004**. The bar misses.

## Training coefficients

Flow is points per within-week standard deviation. Ownership is points per 10 percentage points, at the training-fold average. The standard errors are homoskedastic and are not the gate.

| training leaves out | flow c | se | ownership d | se | interaction e | se |
|---|---:|---:|---:|---:|---:|---:|
| 2022-23 | +0.1386 | 0.0210 | +0.0779 | 0.0242 | -0.0232 | 0.0167 |
| 2023-24 | +0.2027 | 0.0211 | +0.2529 | 0.0222 | +0.0228 | 0.0160 |
| 2024-25 | +0.2124 | 0.0214 | +0.2401 | 0.0220 | -0.0227 | 0.0160 |
| 2025-26 | +0.1772 | 0.0213 | +0.2560 | 0.0222 | -0.0030 | 0.0160 |

## Sample

- 2022-23: 7085 eligible rows, transfer merge 1.0000.
- 2023-24: 7330 eligible rows, transfer merge 1.0000.
- 2024-25: 7427 eligible rows, transfer merge 1.0000.
- 2025-26: 7464 eligible rows, transfer merge 1.0000.
- Eligible weeks that contain two fixtures: 859.
- Correlation of the score with flow: +0.284.
- Correlation of the score with ownership: +0.444.
- Correlation of flow with ownership: +0.121.
Those three correlations are not the test. The partial coefficients are.

`score_xp` stays the published score.

Gemini kept the park ([crowd context](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). The error fell in every left-out season, and the mean fall was 0.011, short of 0.02. The interaction raised the error in every season. The in-sample slopes stay off the score.
