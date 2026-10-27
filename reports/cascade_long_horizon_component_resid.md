# cascade_long vs cascade_short — per-GW E[G,CS,GA] residuals

Season **2025-26** · decision steps GW8–36 · N=40 · goal_weight=`rate` · wall 523.4s

## What this is

- **long** — Opta/oracle λ on **t+2 … end** (product cascade_long).
- **short** — FD centreline λ on **t:t+1** (product cascade_short), same refreshed pools + same L3 XI, no captain.
- Plot = **per-GW** mean resid (Σ resid / n_plan) so horizons compare.

![per-GW shift](/opt/cursor/artifacts/cascade_long_horizon_component_resid_shift.png)

![long target revision](/opt/cursor/artifacts/cascade_long_horizon_component_resid_revision.png)

- Mean per-GW resid_G: long **+3.47**, short **+1.86**
- Mean per-GW resid_CS: long **+1.56**, short **-0.08**

## Per decision GW (per-GW resid)

| t | long n | long G/n | short n | short G/n | long CS/n | short CS/n | long GA/n | short GA/n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 29 | +3.14 | 2 | -4.85 | -0.34 | -9.89 | -0.56 | -1.31 |
| 9 | 28 | +1.44 | 2 | +2.25 | +0.49 | -6.44 | -0.31 | -0.24 |
| 10 | 27 | +3.81 | 2 | +1.48 | +1.60 | -2.27 | -0.49 | +0.76 |
| 11 | 26 | +3.22 | 2 | +8.18 | +1.23 | +4.70 | -0.44 | +0.99 |
| 12 | 25 | +3.29 | 2 | +4.62 | +0.73 | +2.70 | -0.66 | -0.65 |
| 13 | 24 | +6.27 | 2 | +1.93 | +1.62 | +4.89 | -0.71 | +0.45 |
| 14 | 23 | +4.61 | 2 | +1.76 | +0.98 | -3.79 | -1.43 | -0.35 |
| 15 | 22 | +2.70 | 2 | +0.62 | +2.45 | -1.44 | -0.56 | -1.41 |
| 16 | 21 | +3.11 | 2 | -7.33 | +2.46 | -2.85 | -0.31 | -0.66 |
| 17 | 20 | +2.44 | 2 | -0.82 | +1.22 | +2.73 | -0.63 | +1.04 |
| 18 | 19 | +3.84 | 2 | +8.20 | +1.38 | +4.41 | -0.66 | -0.82 |
| 19 | 18 | +4.06 | 2 | -5.56 | +0.62 | +4.47 | -0.43 | -0.35 |
| 20 | 17 | +4.92 | 2 | -11.35 | +2.14 | +1.39 | -0.41 | +0.62 |
| 21 | 16 | +5.74 | 2 | +0.44 | +1.80 | -4.24 | -0.61 | -1.09 |
| 22 | 15 | +4.65 | 2 | +10.26 | +0.66 | -3.74 | -0.69 | -0.47 |
| 23 | 14 | +5.34 | 2 | -1.21 | +2.33 | +2.29 | -0.84 | +1.20 |
| 24 | 13 | +4.71 | 2 | -1.55 | +2.06 | -2.23 | -0.48 | +0.11 |
| 25 | 12 | +4.58 | 2 | +0.72 | +1.68 | +1.36 | -0.29 | -0.76 |
| 26 | 11 | +2.57 | 2 | +8.17 | +1.40 | +5.99 | -0.46 | -0.81 |
| 27 | 10 | +3.57 | 2 | +4.39 | +0.79 | +3.60 | -0.28 | -0.81 |
| 28 | 9 | +4.05 | 2 | +2.50 | +1.09 | +1.70 | -0.24 | -0.41 |
| 29 | 8 | +2.89 | 2 | +8.44 | +1.70 | -3.04 | +0.16 | -0.77 |
| 30 | 7 | +1.68 | 2 | +3.41 | +1.48 | -0.02 | -0.03 | -0.25 |
| 31 | 6 | +1.56 | 2 | -1.96 | +1.28 | +3.67 | -1.03 | +1.28 |
| 32 | 5 | +2.47 | 2 | +1.60 | -1.11 | +1.53 | -0.97 | +1.01 |
| 33 | 4 | +3.33 | 2 | +6.67 | +0.93 | +2.56 | -1.08 | -0.97 |
| 34 | 3 | +2.60 | 2 | -3.01 | +1.94 | -0.88 | -0.49 | -1.32 |
| 35 | 2 | +5.24 | 2 | +6.11 | +3.12 | -4.47 | -0.50 | -0.91 |
| 36 | 1 | -1.10 | 2 | +9.94 | +7.47 | -4.89 | -1.05 | -1.20 |

## Mean per-GW resid by decision block

| block | n | long G/n | short G/n | long CS/n | short CS/n |
| --- | ---: | ---: | ---: | ---: | ---: |
| t=8-12 | 5 | +2.98 | +2.33 | +0.74 | -2.24 |
| t=13-19 | 7 | +3.86 | -0.17 | +1.53 | +1.20 |
| t=20-26 | 7 | +4.64 | +0.78 | +1.73 | +0.12 |
| t=27-33 | 7 | +2.79 | +3.58 | +0.88 | +1.43 |
| t=34-36 | 3 | +2.25 | +4.35 | +4.18 | -3.41 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_horizon_component_resid --n 40 --goal-weight rate --steps 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36 --horizon-end 38 --reuse-long /opt/cursor/artifacts/cascade_long_horizon_component_resid_longonly.json
```

Cells: `/opt/cursor/artifacts/cascade_long_horizon_component_resid_cells.json`

