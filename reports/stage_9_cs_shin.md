# Stage 9 — Shin / OU + roll3 → clean sheets

Fixture markets (Shin 1X2 / under 2.5) and leakage-free roll-3 CS rates vs realised team / player clean sheets.

- Player-match rows (joined): **11498**
- Team×fixture rows: **760**
- Player backtest: GKP+DEF, ≥ 3 prior apps; starters = minutes ≥ 60

## Team grain (market → team CS)

| pair | n | MAE | R² | corr | decile_corr | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| p_under → team CS | 760 | 0.472 | -0.210 | 0.063 | 0.513 | 0.255 | 0.452 |
| p_not_lose → team CS | 760 | 0.460 | -0.283 | 0.249 | 0.863 | 0.255 | 0.500 |
| 1−p_lose → team CS | 760 | 0.519 | -0.669 | 0.250 | 0.905 | 0.255 | 0.620 |
| 1−defend_threat → team CS | 760 | 0.460 | -0.283 | 0.249 | 0.863 | 0.255 | 0.500 |
| roll3 team CS → team CS | 760 | 0.378 | -0.321 | 0.027 | nan | 0.255 | 0.264 |

## Player grain — GKP+DEF, minutes ≥ 60

| pair | n | MAE | R² | corr | decile_corr | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| p_under → CS | 3346 | 0.474 | -0.201 | 0.039 | 0.406 | 0.261 | 0.450 |
| p_not_lose → CS | 3346 | 0.454 | -0.235 | 0.275 | 0.891 | 0.261 | 0.497 |
| 1−p_lose → CS | 3346 | 0.511 | -0.605 | 0.274 | 0.918 | 0.261 | 0.617 |
| 1−defend_threat → CS | 3346 | 0.454 | -0.235 | 0.275 | 0.891 | 0.261 | 0.497 |
| roll3 player CS → CS | 3346 | 0.371 | -0.295 | 0.016 | nan | 0.261 | 0.238 |
| roll3 team CS → CS | 3346 | 0.375 | -0.276 | 0.037 | nan | 0.261 | 0.255 |

## By position (mins ≥ 60) — best market vs roll3

| pos | pair | n | corr | decile_corr | R² |
|---|---|---:|---:|---:|---:|
| GKP | p_under → CS | 648 | 0.045 | 0.422 | -0.224 |
| GKP | p_not_lose → CS | 648 | 0.257 | 0.867 | -0.298 |
| GKP | roll3 player CS → CS | 648 | 0.023 | 0.541 | -0.301 |
| DEF | p_under → CS | 2698 | 0.038 | 0.398 | -0.196 |
| DEF | p_not_lose → CS | 2698 | 0.280 | 0.891 | -0.220 |
| DEF | roll3 player CS → CS | 2698 | 0.015 | nan | -0.294 |

## Plots

- `data/plots/cs_shin_team.png`
- `data/plots/cs_shin_player.png`

## Output

- `data/processed/cs_shin_backtest.csv`
- `data/processed/cs_shin_features.csv`

## Read

- Markets are **not** on CS-probability scale → prefer **corr / decile_corr** over R²/MAE.
- If market corr ≫ roll3 CS, keep Shin/OU as the CS channel prior.
- If both weak (corr ≲ 0.15), CS is mostly noise given current features.
