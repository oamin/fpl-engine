# Stage 21 — Smoothed Ridge + dual portfolio/XI

**1.** Ridge retrained on **forward mean points** `y = mean(pts_t, pts_{t+1}, pts_{t+2})` (within season/player).

**2.** Transfer **hysteresis:** `V − 1.0×n_transfers` plus hold unless ΔV ≥ 1.25.

**3.** **dual_xp_ridge:** xP for GW1 build + swaps; Ridge for XI/captain.

- Horizon V for swaps: H=3 (unchanged from stage 19b)
- GWs: **5–38** (n=34)

## Final standings (captain ×2 − hits)

| method | total | mean/GW | vs ridge_h3 | vs stage19 xp_ft |
|---|---:|---:|---:|---:|
| xp_budget | 2116 | 62.2 | +302 | — |
| ridge_h3_ft | 1814 | 53.4 | +0 | — |
| xp_ft | 1800 | 52.9 | -14 | -48 |
| dual_xp_ridge_ft | 1755 | 51.6 | -59 | — |
| ridge_1gw_ft | 1732 | 50.9 | -82 | -14 |

## Transfer behaviour

- ridge_h3_ft: tx=0.47, holds=27/34, hits=0
- ridge_1gw_ft: tx=0.97, holds=10/34, hits=1
- xp_ft: tx=0.97, holds=6/34, hits=2
- dual_xp_ridge_ft: tx=0.97, holds=6/34, hits=2

## Key deltas

- ridge_h3 vs ridge_1gw: **+82**
- ridge_h3 vs xp_ft: **+14**
- dual vs xp_ft: **-45**
- dual vs ridge_1gw: **+23**
- Best FT vs xp_budget (2116): **-302**

## Gate verdict

**PASS — smoothed Ridge leads FT methods (h3 vs xp +14, vs 1gw +82)**

## Plots

- `data/plots/season_climb_stage21.png`

## Output

- `data/processed/season_climb_stage21.csv`
