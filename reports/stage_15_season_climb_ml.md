# Stage 15 — Walk-forward ML vs xP / exp_points (season climb)

Same stripped XI climb as stage 14. ML scores are **walk-forward**: train on all player-GWs with `gw < t`, predict GW `t` points, never see the future.

- ML warm-up until ≥ **600** training rows
- Evaluated GWs: **7–38** (n=32) — baselines restricted to the same window
- Features: `xmi, exp_points, roll3_points, exp_xG, exp_xA, share_xG, share_xA, lam_scored, …` + position one-hot (+ xP components)

## Models

| name | model |
|---|---|
| `ml_ridge` | StandardScaler + Ridge(α=5) |
| `ml_hgb` | sklearn HistGradientBoosting |
| `ml_lgbm` | LightGBM regressor |
| `ml_resid` | HGB on (points−xP), score = xP + residual̂ |

## Final standings (captain ×2, aligned GWs)

| method | total | mean/GW | vs exp_points | vs xP |
|---|---:|---:|---:|---:|
| ml_ridge | 1967 | 61.5 | +244 | +38 |
| xp | 1929 | 60.3 | +206 | +0 |
| blend_xp_exp | 1909 | 59.7 | +186 | -20 |
| price | 1841 | 57.5 | +118 | -88 |
| ml_lgbm | 1818 | 56.8 | +95 | -111 |
| ml_resid | 1810 | 56.6 | +87 | -119 |
| ml_hgb | 1794 | 56.1 | +71 | -135 |
| exp_points | 1723 | 53.8 | +0 | -206 |
| roll3_points | 1592 | 49.8 | -131 | -337 |
| xmi | 1499 | 46.8 | -224 | -430 |
| random | 1194 | 37.3 | -529 | -735 |

## vs exp_points by GW

| method | GWs ahead | GWs behind | mean Δ/GW |
|---|---:|---:|---:|
| ml_ridge | 22 | 10 | +7.62 |
| xp | 21 | 10 | +6.44 |
| blend_xp_exp | 22 | 8 | +5.81 |
| price | 20 | 12 | +3.69 |
| ml_lgbm | 17 | 11 | +2.97 |
| ml_resid | 20 | 12 | +2.72 |
| ml_hgb | 15 | 16 | +2.22 |
| roll3_points | 14 | 18 | -4.09 |
| xmi | 9 | 23 | -7.00 |
| random | 7 | 25 | -16.53 |

## Gate verdict

**PASS — ml_ridge beats both (vs xP +38, vs exp +244)**

Pass bar: best ML finishes ≥ **+10** vs both `exp_points` and `xp` on the aligned GW window.

## Plots

- `data/plots/season_climb_ml.png`

## Output

- `data/processed/season_climb_ml.csv`
