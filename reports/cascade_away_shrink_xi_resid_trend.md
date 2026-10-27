# Opta XI resid G/CS vs away_shrink

2025-26 GW8–38 · π∝r · Opta λ · N=40 · shared L3 XI · wall 1056s

Formula: `λ_H` unchanged; `λ_A ← λ_A · s`. Default `s=1.0` unchanged.

![trend](/opt/cursor/artifacts/cascade_away_shrink_xi_resid_trend.png)

| s | resid G | resid CS | resid GA | MAE_G | MAE_CS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | -38.6 | +53.3 | +5.6 | 1.084 | 0.934 |
| 0.55 | -33.8 | +43.3 | +3.8 | 1.095 | 0.922 |
| 0.60 | -25.6 | +31.1 | +2.1 | 1.110 | 0.910 |
| 0.65 | -18.8 | +20.3 | +0.3 | 1.120 | 0.909 |
| 0.70 | -9.4 | +10.5 | -1.5 | 1.141 | 0.897 |
| 0.75 | -2.4 | +2.4 | -3.8 | 1.151 | 0.889 |
| 0.80 | +5.4 | -3.1 | -5.7 | 1.152 | 0.882 |
| 0.85 | +9.5 | -10.2 | -7.3 | 1.159 | 0.874 |
| 0.86 | +12.1 | -11.1 | -7.7 | 1.162 | 0.873 |
| 0.90 | +19.5 | -16.5 | -9.0 | 1.171 | 0.871 |
| 0.95 | +29.7 | -24.4 | -11.5 | 1.193 | 0.872 |
| 1.00 | +32.5 | -29.2 | -14.0 | 1.205 | 0.873 |

- Closest resid G to 0: **s=0.75** (-2.4).
- Closest resid CS to 0: **s=0.75** (+2.4).

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_away_shrink_xi_resid_trend
```

