# Monte Carlo Shin scoreline + player goal allocation (2025/26)

Opening 1X2 (and OU 2.5) → Shin de-vig → λ_h / λ_a → MC scorelines conditioned on Shin W/D/A → goals allocated to squad players by expanding xG prior (2024/25 seed) with position floors FWD>MID>DEF>GKP.

- Fixtures: **380**
- Sims per fixture: **1000**
- NB/Gamma rate draws: **True**

## Match scoreline residuals (home − away)

| metric | value |
|---|---:|
| mean GD residual (act − E) | 0.0092 |
| MAE GD residual | 1.2168 |
| RMSE GD residual | 1.5068 |
| corr(E[GD], actual GD) | 0.3892 |
| MAE home-goals residual | 0.9153 |
| MAE away-goals residual | 0.8393 |

## Player goal residuals (minutes > 0)

| metric | value |
|---|---:|
| n player–matches | 11498 |
| mean residual | 0.0038 |
| MAE residual | 0.1461 |
| RMSE residual | 0.3022 |
| corr(E[goals], actual) | 0.2571 |
| Σ goals act / exp | 1009.0 / 965.3 |

## By position

| pos | n | mean resid | MAE | corr(E, act) |
|---|---:|---:|---:|---:|
| DEF | 3950 | -0.0029 | 0.0691 | 0.0678 |
| FWD | 1434 | 0.0423 | 0.3319 | 0.1899 |
| GKP | 767 | -0.0128 | 0.0128 | None |
| MID | 5347 | 0.0008 | 0.1722 | 0.1831 |

## Artefacts

- `data/processed/mc_shin_match_residuals_2025_26.csv`
- `data/processed/mc_shin_player_residuals_2025_26.csv`
- `reports/figures/mc_shin_player_goals_2025_26.json`

Reproduce: `python3 -m src.models.mc_shin_player_goals`
