# Mean xi_ft run — count residual E[G, CS, GA] − actual

Source artifact: `/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json`
Season **2025-26** · N=40 · seed=7 · wall 285.3s

## Share pool (default)

Attack dice / π on **full club ``m>0``**; selection stays ≥60′. Shares are not re-normalised on the thin board (`protocol.cascade_short_generative.share_pool`).

- Prior thin-share OL Σ resid G: **+48.45**
- Full-share OL Σ resid G: **+11.77** (Δ -36.68)

## Own goals / non-attributed goals

Generative attack puts mass on ``∅_G``: each rolled match goal is attributed to a player with probability ``α_G``, else ``∅_G`` (own goals / non-FPL scorers). ``α_G`` is the expanding player-goals / match-goals rate as-of the decision GW.

- OL as-of α_G = **0.9556** (⇒ empty mass 1−α_G = **0.0444**)
- Realised empty rate on OL strands: **0.0454** (647 / 14265 goals)

Counts below use **FPL-attributed** E[G] (∅_G excluded). CS = eligible clean sheet ``(m≥60)∧(GA=0)``. GA = team goals against when ``m>0``.

![per-match residuals](/opt/cursor/artifacts/cascade_mean_xi_ft_count_resid.png)

## Open-loop XI (GW8 boards → GW8–19)

- Player-GWs: **132**

| count | Σ E | Σ actual | Σ resid | mean resid / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 39.77 | 28.00 | +11.77 | +0.089 | 0.346 |
| CS | 47.55 | 34.00 | +13.55 | +0.103 | 0.421 |
| GA | 141.78 | 126.00 | +15.78 | +0.120 | 0.787 |

### Per GW (residual / match on XI)

| GW | n | r_G | r_CS | r_GA |
| --- | ---: | ---: | ---: | ---: |
| 8 | 11 | -0.043 | -0.273 | +0.705 |
| 9 | 11 | +0.189 | +0.089 | -0.170 |
| 10 | 11 | -0.011 | +0.082 | +0.139 |
| 11 | 11 | -0.002 | +0.014 | -0.111 |
| 12 | 11 | +0.348 | +0.223 | -0.032 |
| 13 | 11 | +0.373 | +0.268 | +0.098 |
| 14 | 11 | -0.116 | +0.352 | -0.470 |
| 15 | 11 | -0.002 | +0.127 | +0.323 |
| 16 | 11 | +0.014 | -0.043 | +0.173 |
| 17 | 11 | -0.014 | -0.289 | +0.764 |
| 18 | 11 | +0.068 | +0.386 | -0.198 |
| 19 | 11 | +0.268 | +0.295 | +0.216 |

## Rolling committed XI (step boards → GW8–10)

- Player-GWs: **33**

| count | Σ E | Σ actual | Σ resid | mean resid / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 9.10 | 9.00 | +0.10 | +0.003 | 0.377 |
| CS | 12.80 | 14.00 | -1.20 | -0.036 | 0.406 |
| GA | 32.68 | 29.00 | +3.68 | +0.111 | 0.686 |

### Per GW (residual / match on XI)

| GW | n | r_G | r_CS | r_GA |
| --- | ---: | ---: | ---: | ---: |
| 8 | 11 | -0.043 | -0.273 | +0.705 |
| 9 | 11 | +0.152 | +0.086 | -0.264 |
| 10 | 11 | -0.100 | +0.077 | -0.107 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_count_resid --artifact /opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json --n 40
```

