# Mean xi_ft run — G / CS / GC residual (xP − actual)

Source artifact: `/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json`
Season **2025-26** · N=100 · seed=7 · wall 439.8s

xP = generative mean-board component (`xp_goals` / `xp_cs` / `xp_gc`). Actual P from FPL goals / CS (≥60′ & 0 GA) / GC bins (−1 per 2 for GKP/DEF). Residual = **xP − P** on the planned XI. Charts use **residual / match** = GW Σ resid ÷ XI player-matches.

Plot: `/opt/cursor/artifacts/cascade_mean_xi_ft_component_resid.png`

## Open-loop XI (GW8 boards → GW8–19)

- Player-GWs: **132**

| component | Σ xP | Σ actual P | Σ resid | mean resid / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 357.2 | 127.0 | +230.2 | +1.744 | 2.371 |
| CS | 95.1 | 73.0 | +22.1 | +0.167 | 0.837 |
| GC | -15.4 | -10.0 | -5.4 | -0.041 | 0.137 |
| **G+CS+GC** | 437.0 | 190.0 | +247.0 |  | 3.346 |

### Per GW (residual / match on XI)

| GW | n | r_G / match | r_CS / match | r_GC / match |
| --- | ---: | ---: | ---: | ---: |
| 8 | 11 | +1.12 | -0.83 | -0.07 |
| 9 | 11 | +1.47 | -0.44 | -0.09 |
| 10 | 11 | +1.52 | -0.09 | -0.08 |
| 11 | 11 | +1.37 | +0.48 | +0.24 |
| 12 | 11 | +2.56 | +0.66 | -0.02 |
| 13 | 11 | +3.21 | +0.52 | -0.04 |
| 14 | 11 | +1.01 | +0.61 | -0.12 |
| 15 | 11 | +1.38 | +0.26 | -0.04 |
| 16 | 11 | +1.79 | -0.10 | -0.10 |
| 17 | 11 | +1.62 | -0.59 | -0.17 |
| 18 | 11 | +1.47 | +0.80 | +0.03 |
| 19 | 11 | +2.39 | +0.74 | -0.02 |

## Rolling committed XI (step boards → GW8–10)

- Player-GWs: **33**

| component | Σ xP | Σ actual P | Σ resid | mean resid / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 80.4 | 40.0 | +40.4 | +1.225 | 2.014 |
| CS | 25.9 | 42.0 | -16.1 | -0.487 | 0.838 |
| GC | -2.3 | 0.0 | -2.3 | -0.070 | 0.070 |
| **G+CS+GC** | 104.1 | 82.0 | +22.1 |  | 2.922 |

### Per GW (residual / match on XI)

| GW | n | r_G / match | r_CS / match | r_GC / match |
| --- | ---: | ---: | ---: | ---: |
| 8 | 11 | +1.12 | -0.83 | -0.07 |
| 9 | 11 | +1.58 | -0.32 | -0.07 |
| 10 | 11 | +0.98 | -0.31 | -0.07 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_component_resid --artifact /opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json --n 100
```

