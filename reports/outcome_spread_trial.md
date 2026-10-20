# Outcome model trial — μ (raw) + σ_out + blank/haul

Diagnostic only (not a gated forecast screen). Pool: minutes≥60, n_out≥3, μ = **score_xp_v2** (non-shrunk).

- Blank = points ≤ **2**; haul = points ≥ **10**
- Mixture blank level = 1; haul level = max(P90, 12, μ+σ)

## 1. Non-shrunk μ + Gaussian σ_out

- n = **22358**
- μ bias (y−μ): **+0.093**; MAE **2.323**
- z-SD: **1.320** (target ≈1); 1σ coverage **73.5%**; 2σ **92.1%**
- Gaussian mean P(blank) **0.329** vs realised **0.575**
- Gaussian mean P(haul) **0.040** vs realised **0.060**

## 2. Empirical blank / haul rates (as-of)

- blank_rate: pred mean **0.572**, realised **0.575**, Brier **0.2559**
- haul_rate: pred mean **0.060**, realised **0.060**, Brier **0.0600**

### Blank-rate deciles

| bin | n | pred | realised |
| --- | ---: | ---: | ---: |
| (-0.001, 0.333] | 3178 | 0.247 | 0.476 |
| (0.333, 0.4] | 1626 | 0.382 | 0.479 |
| (0.4, 0.471] | 1913 | 0.441 | 0.507 |
| (0.471, 0.516] | 2228 | 0.498 | 0.556 |
| (0.516, 0.571] | 2371 | 0.549 | 0.542 |
| (0.571, 0.625] | 2244 | 0.603 | 0.592 |
| (0.625, 0.667] | 2361 | 0.659 | 0.603 |
| (0.667, 0.75] | 2654 | 0.724 | 0.635 |
| (0.75, 0.833] | 1791 | 0.802 | 0.687 |
| (0.833, 1.0] | 1992 | 0.939 | 0.711 |

### Haul-rate deciles

| bin | n | pred | realised |
| --- | ---: | ---: | ---: |
| (-0.001, 0.0476] | 13482 | 0.003 | 0.044 |
| (0.0476, 0.08] | 2206 | 0.065 | 0.064 |
| (0.08, 0.125] | 2623 | 0.104 | 0.070 |
| (0.125, 0.188] | 1861 | 0.156 | 0.095 |
| (0.188, 0.8] | 2186 | 0.275 | 0.113 |

## 3. Scenario arms (Monte Carlo)

n_draw = 200. Lower Brier is better for blank/haul hits.

| arm | mean bias | MAE | P(blank) | blank Brier | P(haul) | haul Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Realised | — | — | 0.575 | — | 0.060 | — |
| Empirical rates (mean) | — | — | 0.572 | 0.2559 | 0.060 | 0.0600 |
| Gaussian N(μ,σ) | +0.094 | 2.325 | 0.329 | 0.2955 | 0.040 | 0.0565 |
| Mixture (rates+N) | +0.904 | 2.310 | 0.684 | 0.2558 | 0.076 | 0.0622 |

## By season

| season | n | μ bias | z-SD | 1σ cov | blank Brier (rates) | haul Brier (rates) | Δ blank Brier (mix−gauss) | Δ haul Brier (mix−gauss) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 5542 | +0.122 | 1.24 | 73.4% | 0.2561 | 0.0595 | -0.0488 | +0.0050 |
| 2023-24 | 5651 | +0.083 | 1.41 | 73.8% | 0.2484 | 0.0603 | -0.0543 | +0.0051 |
| 2024-25 | 5587 | +0.107 | 1.34 | 73.5% | 0.2503 | 0.0546 | -0.0415 | +0.0056 |
| 2025-26 | 5578 | +0.062 | 1.28 | 73.1% | 0.2688 | 0.0654 | -0.0170 | +0.0076 |

## Verdict

Raw μ bias +0.09 pts; z-SD 1.32 (off target — σ_out mis-scaled). Empirical blank_rate Brier 0.2559 vs Gaussian-implied 0.2948 (rates win). Empirical haul_rate Brier 0.0600 vs Gaussian-implied 0.0564 (Gaussian competitive). Mixture vs Gaussian blank Brier Δ -0.0397; haul Δ +0.0057 (mixture helps tails).

JSON: `data/processed/outcome_spread_trial.json`

