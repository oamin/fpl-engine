# Stage 16 — Multi-season walk-forward Ridge (season climb)

Natural next step after stage 15: train Ridge on **prior seasons + earlier 2025/26 GWs**, predict each 2025/26 GW, run the locked stripped XI climb.

- Train seasons: 2022-23, 2023-24, 2024-25, 2025-26
- Eval: **2025-26** GWs **7–38** (n=32)
- Min train rows: **600**
- Rows by season: 2022-23=9776, 2023-24=9765, 2024-25=9968, 2025-26=9965

Within-season expanding priors only (player keys are `season:element`). DefCon features are 0 before 2025/26.

## Final standings (captain ×2, aligned GWs)

| method | total | mean/GW | vs exp | vs xP | vs ridge_1s |
|---|---:|---:|---:|---:|---:|
| ridge_ms | 2022 | 63.2 | +299 | +93 | +55 |
| ridge_1s | 1967 | 61.5 | +244 | +38 | +0 |
| xp | 1929 | 60.3 | +206 | +0 | -38 |
| blend_xp_exp | 1909 | 59.7 | +186 | -20 | -58 |
| price | 1841 | 57.5 | +118 | -88 | -126 |
| exp_points | 1723 | 53.8 | +0 | -206 | -244 |
| roll3_points | 1592 | 49.8 | -131 | -337 | -375 |
| xmi | 1499 | 46.8 | -224 | -430 | -468 |
| random | 1091 | 34.1 | -632 | -838 | -876 |

## Gate verdict

**PASS — ridge_ms best (vs 1-season +55, vs xP +93, vs exp +299)**

## Plots

- `data/plots/season_climb_ridge_ms.png`

## Output

- `data/processed/season_climb_ridge_ms.csv`
