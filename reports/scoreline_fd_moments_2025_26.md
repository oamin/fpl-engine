# Scoreline moments vs actuals — 2025-26

Football-data **opening** AH + OU → Dixon–Coles `M` (BTTS weight 0).
Sides stacked (home and away). Actual CS = opponent scored 0; GC events = floor(GA/2).

- Matches fitted: **379** (758 side-rows)
- Fit ok rate: **1.000** · mean SSE(AH,OU): **1.72e-11**

## xG vs goals
- Pearson **0.296** · Spearman **0.282**
- MAE **0.891** · RMSE **1.094** (constant MAE 0.949)
- Mean pred/actual **1.441 / 1.375** (bias +0.066)

## xCS vs clean sheet
- Pearson **0.243** · Brier **0.1787** (constant 0.1898) · logloss **0.5363**
- Mean pred/actual **0.260 / 0.255** (bias +0.006)

## xGC vs GC events
- Pearson **0.241** · Spearman **0.234**
- MAE **0.495** · RMSE **0.564** (constant MAE 0.533)
- Mean pred/actual **0.491 / 0.454** (bias +0.037)

