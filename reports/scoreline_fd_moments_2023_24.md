# Scoreline moments vs actuals — 2023-24

Football-data **opening** AH + OU → Dixon–Coles `M` (BTTS weight 0).
Sides stacked (home and away). Actual CS = opponent scored 0; GC events = floor(GA/2).

- Matches fitted: **380** (760 side-rows)
- Fit ok rate: **1.000** · mean SSE(AH,OU): **2.30e-11**

## xG vs goals
- Pearson **0.423** · Spearman **0.414**
- MAE **0.956** · RMSE **1.207** (constant MAE 1.081)
- Mean pred/actual **1.568 / 1.639** (bias -0.072)

## xCS vs clean sheet
- Pearson **0.253** · Brier **0.1548** (constant 0.1639) · logloss **0.4785**
- Mean pred/actual **0.240 / 0.207** (bias +0.033)

## xGC vs GC events
- Pearson **0.403** · Spearman **0.392**
- MAE **0.521** · RMSE **0.628** (constant MAE 0.606)
- Mean pred/actual **0.553 / 0.580** (bias -0.027)

