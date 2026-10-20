# `score_xp_v2` SE credibility + shrinkage screen

Estimation SE (`xp_se`) is **not** predictive σ for points. Use it to **shrink**
μ toward a position prior before the MILP — do not subtract η·SE from the objective.

## Prior and K (locked method)

- **Prior:** one mean per position within season, estimated on GW < t only (not position × price band).
- **K:** K_pos = τ²_pos / (τ²_pos + mean_SE²_pos) from GW < t; μ_post = prior + K · (μ − prior); σ_model not in K.
- Constant K leaves **within-position order unchanged**; it rescales gains vs
  the −4 hit and chip thresholds. Calibrate those on μ_post; do **not** stack
  the ~0.5 decision-week haircut on top.

### Walk-forward mean K by position

| position | mean K | mean prior | mean τ² | mean SE² |
| --- | ---: | ---: | ---: | ---: |
| DEF | 0.736 | 3.09 | 1.048 | 0.382 |
| FWD | 0.679 | 4.45 | 2.110 | 0.978 |
| GKP | 0.841 | 2.71 | 0.492 | 0.093 |
| MID | 0.655 | 3.70 | 1.609 | 0.784 |

### Snapshot (2025-26 last walk-forward GW, as-of)

| position | K | prior | τ² |
| --- | ---: | ---: | ---: |
| DEF | 0.778 | 3.61 | 1.067 |
| FWD | 0.731 | 4.17 | 1.631 |
| GKP | 0.884 | 2.70 | 0.444 |
| MID | 0.589 | 3.96 | 0.787 |

## Calibration (walk-forward GW≥5)

- Raw SE: mean z-SD **1.63**, 1σ coverage **53.4%**
- SE + σ_model: mean z-SD **0.97**, 1σ coverage **78.0%**
- Mean σ_model² (from prior weeks): **6.235**

Target z-SD ≈ 1. Raw SE is too narrow; single-GW residual σ_model brings z-SD≈1
but is **not** folded into K. Multi-week σ_model (5–10 GW windows, Poisson noise
subtracted) stays pending.

## Arms (as-of; Gaussian log-score σ=3)

| arm | mean log-score | Δ vs none | MSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| constant | -2.54786 | +0.01342 | 9.546 | 2.313 |
| se | -2.54988 | +0.01140 | 9.582 | 2.276 |
| se_rescaled | -2.55985 | +0.00143 | 9.761 | 2.311 |
| none | -2.56128 | +0.00000 | 9.787 | 2.312 |

## Size and uncertainty (constant − none)

- Paired GW mean Δlog: **+0.01342** [+0.01074, +0.01632] (cluster bootstrap B=1000, seed=0)
- Covers zero: **False**

| season | mean Δlog |
| --- | ---: |
| 2022-23 | +0.01126 |
| 2023-24 | +0.01146 |
| 2024-25 | +0.01396 |
| 2025-26 | +0.01693 |

### Leave-one-season-out (constant − none)

| held out | mean Δlog | 95% CI | covers 0 |
| --- | ---: | --- | --- |
| 2022-23 | +0.01412 | [+0.01093, +0.01742] | False |
| 2023-24 | +0.01408 | [+0.01089, +0.01741] | False |
| 2024-25 | +0.01324 | [+0.01028, +0.01639] | False |
| 2025-26 | +0.01224 | [+0.00939, +0.01516] | False |

### Other paired intervals

- `se_minus_constant`: -0.00202 [-0.00325, -0.00080] (covers 0: False)
- `se_rescaled_minus_constant`: -0.01199 [-0.01473, -0.00902] (covers 0: False)
- `se_minus_none`: +0.01140 [+0.00868, +0.01453] (covers 0: False)

**Verdict:** use mode `constant`. Constant position K beats raw μ; SE-dependent and se_rescaled add nothing beyond constant (dropped).
Dropped arms: se_rescaled, se.

## SE tercile slopes (y on μ)

| tercile | n | mean SE | slope | corr |
| --- | ---: | ---: | ---: | ---: |
| low | 9190 | 0.146 | 0.844 | 0.219 |
| mid | 9189 | 0.434 | 0.732 | 0.203 |
| high | 9189 | 0.926 | 0.551 | 0.206 |

## Implication for Layer 2

- Shrinkage runs **before** the solve; MILP on μ_post (`mode=constant`).
- Forecast credibility is this screen (realised points). The L2 audit is a
  **solver legality smoke** on closed μ boards (as-of moments GW < t) — it does
  not re-prove the forecast gain.
- When transfers/chips land: μ_post everywhere in gain accounting; no double haircut.

JSON: `data/processed/se_credibility.json`

