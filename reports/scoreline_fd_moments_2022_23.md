# Scoreline moments vs actuals — 2022-23

Football-data **opening** AH + OU → Dixon–Coles `M` (BTTS weight 0).
Sides stacked (home and away). Actual CS = opponent scored 0; GC events = floor(GA/2).

- Matches fitted: **380** (760 side-rows)
- Fit ok rate: **1.000** · mean SSE(AH,OU): **2.11e-11**

## xG vs goals
- Pearson **0.377** · Spearman **0.336**
- MAE **0.949** · RMSE **1.225** (constant MAE 1.056)
- Mean pred/actual **1.416 / 1.426** (bias -0.010)

## xCS vs clean sheet
- Pearson **0.240** · Brier **0.1868** (constant 0.1982) · logloss **0.5545**
- Mean pred/actual **0.271 / 0.272** (bias -0.002)

## xGC vs GC events
- Pearson **0.355** · Spearman **0.317**
- MAE **0.522** · RMSE **0.633** (constant MAE 0.589)
- Mean pred/actual **0.481 / 0.491** (bias -0.010)

