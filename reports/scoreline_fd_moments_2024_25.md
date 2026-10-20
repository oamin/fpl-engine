# Scoreline moments vs actuals — 2024-25

Football-data **opening** AH + OU → Dixon–Coles `M` (BTTS weight 0).
Sides stacked (home and away). Actual CS = opponent scored 0; GC events = floor(GA/2).

- Matches fitted: **380** (760 side-rows)
- Fit ok rate: **1.000** · mean SSE(AH,OU): **2.30e-11**

## xG vs goals
- Pearson **0.356** · Spearman **0.342**
- MAE **0.921** · RMSE **1.160** (constant MAE 0.998)
- Mean pred/actual **1.562 / 1.467** (bias +0.095)

## xCS vs clean sheet
- Pearson **0.221** · Brier **0.1708** (constant 0.1794) · logloss **0.5188**
- Mean pred/actual **0.237 / 0.234** (bias +0.002)

## xGC vs GC events
- Pearson **0.328** · Spearman **0.315**
- MAE **0.518** · RMSE **0.610** (constant MAE 0.578)
- Mean pred/actual **0.550 / 0.509** (bias +0.040)

