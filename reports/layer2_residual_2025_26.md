# Layer 2 residual audit — 2025-26

Walk-forward constant-K μ_post (GW < t moments, same as `se_credibility`).
Residual = actual `total_points` − predicted.

Player-GWs: **6664** · Gameweeks: **33** (GW≥5).

## Overall

| metric | raw μ | μ_post | Δ (post − raw) |
| --- | ---: | ---: | ---: |
| bias | +0.1004 | +0.1057 | +0.0053 |
| MAE | 2.3653 | 2.3494 | -0.0159 |
| RMSE | 3.1630 | 3.1147 | -0.0482 |
| MSE | 10.0043 | 9.7016 | -0.3027 |
| log-score (σ=3) | -2.57334 | -2.55653 | +0.01682 |
| corr(pred, y) | 0.1963 | 0.1972 | +0.0010 |
| slope y~pred | 0.5115 | 0.7317 | +0.2203 |
| resid SD | 3.1616 | 3.1132 | -0.0484 |
| P(|resid|≤2) | 53.2% | 53.2% | — |
| P(|resid|≤4) | 85.0% | 86.4% | — |

## Paired by gameweek

- Mean Δlog (post − raw): **+0.01693**
- Mean ΔMSE (post − raw): **-0.3047** (negative = post better)
- GWs where post wins on log-score: **78.8%**
- GWs where post wins on MSE: **78.8%**

## By position

| pos | n | MAE raw | MAE post | MSE raw | MSE post | Δlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DEF | 2600 | 2.470 | 2.462 | 10.36 | 10.12 | +0.01304 |
| FWD | 662 | 2.759 | 2.785 | 12.76 | 12.27 | +0.02707 |
| GKP | 654 | 1.905 | 1.892 | 7.48 | 7.40 | +0.00445 |
| MID | 2748 | 2.281 | 2.247 | 9.61 | 9.23 | +0.02087 |

## Verdict

μ_post improves both log-score and MSE vs raw μ on 2025-26 walk-forward.

Lower MAE/MSE and higher log-score favour μ_post. Slope closer to 1 means less
optimiser's-curse compression of realised-vs-predicted.

JSON: `data/processed/layer2_residual_2025_26.json`

