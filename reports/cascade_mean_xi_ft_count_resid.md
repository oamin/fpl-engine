# Mean xi_ft run — count residual E[G, CS, GA] − actual

Source artifact: `/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json`
Season **2025-26** · N=100 · seed=7 · wall 437.4s

## Own goals / non-attributed goals

Generative attack puts mass on ``∅_G``: each rolled match goal is attributed to a player with probability ``α_G``, else ``∅_G`` (own goals / non-FPL scorers). ``α_G`` is the expanding player-goals / match-goals rate as-of the decision GW.

- OL as-of α_G = **0.9556** (⇒ empty mass 1−α_G = **0.0444**)
- Realised empty rate on OL strands: **0.0450** (1618 / 35988 goals)

Counts below use **FPL-attributed** E[G] (∅_G excluded). CS = eligible clean sheet ``(m≥60)∧(GA=0)``. GA = team goals against when ``m>0``.

![per-match residuals](/opt/cursor/artifacts/cascade_mean_xi_ft_count_resid.png)

## Open-loop XI (GW8 boards → GW8–19)

- Player-GWs: **132**

| count | Σ E | Σ actual | Σ resid | mean resid / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 76.45 | 28.00 | +48.45 | +0.367 | 0.496 |
| CS | 47.58 | 34.00 | +13.58 | +0.103 | 0.419 |
| GA | 141.67 | 126.00 | +15.67 | +0.119 | 0.791 |

### Per GW (residual / match on XI)

| GW | n | r_G | r_CS | r_GA |
| --- | ---: | ---: | ---: | ---: |
| 8 | 11 | +0.216 | -0.275 | +0.725 |
| 9 | 11 | +0.352 | +0.073 | -0.185 |
| 10 | 11 | +0.276 | +0.067 | +0.187 |
| 11 | 11 | +0.274 | +0.030 | -0.135 |
| 12 | 11 | +0.550 | +0.255 | -0.041 |
| 13 | 11 | +0.685 | +0.250 | +0.110 |
| 14 | 11 | +0.198 | +0.364 | -0.512 |
| 15 | 11 | +0.322 | +0.148 | +0.343 |
| 16 | 11 | +0.360 | -0.046 | +0.185 |
| 17 | 11 | +0.311 | -0.291 | +0.760 |
| 18 | 11 | +0.315 | +0.361 | -0.205 |
| 19 | 11 | +0.545 | +0.300 | +0.192 |

## Rolling committed XI (step boards → GW8–10)

- Player-GWs: **33**

| count | Σ E | Σ actual | Σ resid | mean resid / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 17.63 | 9.00 | +8.63 | +0.262 | 0.429 |
| CS | 12.36 | 14.00 | -1.64 | -0.050 | 0.404 |
| GA | 33.67 | 29.00 | +4.67 | +0.142 | 0.702 |

### Per GW (residual / match on XI)

| GW | n | r_G | r_CS | r_GA |
| --- | ---: | ---: | ---: | ---: |
| 8 | 11 | +0.216 | -0.275 | +0.725 |
| 9 | 11 | +0.368 | +0.085 | -0.255 |
| 10 | 11 | +0.200 | +0.042 | -0.046 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_count_resid --artifact /opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json --n 100
```

