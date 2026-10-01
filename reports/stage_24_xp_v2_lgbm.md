# Stage 24 — xP v2 backtest + LightGBM non-linearity

Refactored equation (GKP goals=10, no `play_scale`, probabilistic appearance, BPS proxy, GC/YC deductions) evaluated on **point-level quality** and the **stripped season climb**. LightGBM learns residual non-linearities walk-forward (`gw < t` only).

- Aligned climb GWs: **7–38** (n=32)
- Train warm-up: ≥ **600** rows

## Equation (v2)

```
xP = appear + goals + assists + cs + defcon + bps − deductions
appear = p_play·1 + p60·1
goals/ast = share × λ × pts   (no play_scale)
bps ≈ 0.18·xp_goals + 0.12·xp_assists + 0.08·xp_cs
deductions = p60·λ_conc/2 (GKP/DEF) + (xMi/90)·0.15
```

## Point-level quality (pooled, n_prior≥3)

| universe | predictor | n | Spearman | MAE | bias | mean_pred | mean_y |
|---|---|---:|---:|---:|---:|---:|---:|
| ALL | xP v2 | 9965 | 0.304 | 2.08 | -0.10 | 2.96 | 3.06 |
| ALL | exp points | 9965 | 0.257 | 2.16 | -0.03 | 3.03 | 3.06 |
| ALL | roll3 points | 9965 | 0.213 | 2.33 | +0.01 | 3.06 | 3.06 |
| GKP | xP v2 | 655 | 0.175 | 1.91 | -0.72 | 2.62 | 3.34 |
| GKP | exp points | 655 | 0.034 | 2.22 | +0.12 | 3.46 | 3.34 |
| GKP | roll3 points | 655 | 0.036 | 2.38 | +0.04 | 3.38 | 3.34 |
| DEF | xP v2 | 3415 | 0.260 | 2.33 | +0.11 | 3.22 | 3.10 |
| DEF | exp points | 3415 | 0.172 | 2.40 | +0.01 | 3.11 | 3.10 |
| DEF | roll3 points | 3415 | 0.115 | 2.59 | +0.02 | 3.13 | 3.10 |
| MID | xP v2 | 4647 | 0.373 | 1.92 | -0.09 | 2.90 | 2.99 |
| MID | exp points | 4647 | 0.320 | 1.98 | -0.06 | 2.92 | 2.99 |
| MID | roll3 points | 4647 | 0.287 | 2.14 | -0.01 | 2.98 | 2.99 |
| FWD | xP v2 | 1248 | 0.390 | 2.04 | -0.38 | 2.65 | 3.02 |
| FWD | exp points | 1248 | 0.342 | 2.17 | -0.09 | 2.93 | 3.02 |
| FWD | roll3 points | 1248 | 0.316 | 2.29 | -0.01 | 3.02 | 3.02 |

## Horizon IC — regulars xMi≥45 (Spearman vs mean pts over next H)

| predictor | H=1 | H=3 | H=5 | H=8 |
|---|---:|---:|---:|---:|
| xP engine | 0.201 | 0.240 | 0.275 | 0.322 |
| exp points | 0.155 | 0.237 | 0.287 | 0.341 |
| roll3 points | 0.110 | 0.139 | 0.155 | 0.202 |
| xMi | 0.130 | 0.178 | 0.204 | 0.231 |

## Walk-forward ML point metrics (aligned GWs)

| score | Spearman | MAE | bias |
|---|---:|---:|---:|
| xp | 0.308 | 2.07 | -0.10 |
| exp_points | 0.257 | 2.16 | -0.05 |
| lgbm_direct | 0.237 | 2.19 | +0.04 |
| lgbm_resid | 0.236 | 2.19 | +0.04 |
| lgbm_nl | 0.250 | 2.17 | +0.03 |

## Season climb (captain ×2, aligned GWs)

| method | total | mean/GW | vs exp | vs xP |
|---|---:|---:|---:|---:|
| blend_xp_exp | 1962 | 61.3 | +239 | +22 |
| xp | 1940 | 60.6 | +217 | +0 |
| lgbm_nl | 1866 | 58.3 | +143 | -74 |
| lgbm_resid | 1862 | 58.2 | +139 | -78 |
| price | 1841 | 57.5 | +118 | -99 |
| lgbm_direct | 1808 | 56.5 | +85 | -132 |
| exp_points | 1723 | 53.8 | +0 | -217 |
| roll3_points | 1592 | 49.8 | -131 | -348 |
| xmi | 1499 | 46.8 | -224 | -441 |
| random | 1194 | 37.3 | -529 | -746 |

## LightGBM residual — top features (last GW fit, gain)

| feature | gain_resid | gain_direct |
|---|---:|---:|
| e_total | 417 | 428 |
| exp_points | 404 | 384 |
| lam_scored | 380 | 352 |
| xp_bps | 322 | 308 |
| xp | 321 | 273 |
| exp_xG | 297 | 304 |
| value | 272 | 292 |
| p_cs_mkt | 267 | 283 |
| xp_goals | 266 | 244 |
| exp_xA | 261 | 264 |
| share_xA | 257 | 263 |
| p_not_lose | 257 | 264 |

## Gate verdicts

- **Quality:** PASS — xP Spearman 0.304 (Δ vs exp +0.048), bias -0.10
- **Climb + LGBM:** PARTIAL — lgbm_nl beats exp (+143) but trails xP (-74)

Pass bar: xP Spearman ≥ exp on pooled appearances with |bias| modest; best LGBM finishes ≥ **+10** vs both exp and xP on the climb.

## Plots / outputs

- `data/plots/xp_v2_lgbm_gate.png`
- `data/processed/xp_v2_quality.csv`
- `data/processed/xp_v2_lgbm_climb.csv`
- `data/processed/xp_v2_lgbm_importance.csv`
