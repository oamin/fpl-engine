# Stage 22 — Fixture-aware horizon + structural 2-transfers

On top of stage-21 **ridge_h3** + switch penalty:

1. **Fixture-aware V:** scale frozen score by Vaastav fixture count (0 blank / 1 SGW / 2 DGW); horizon **H=4**
2. **Structural 2-transfers:** simultaneous sells/buys preserving 2/5/5/3 (fund premium via cross-pos pairs)
3. Hold unless ΔV ≥ 1.25; switch penalty 1.0/tx

- GWs: **5–38** (n=34)
- Stage-21 ridge_h3 baseline: **1814**

## Final standings (captain ×2 − hits)

| method | total | mean/GW | vs ridge_h3_ft | vs stage21 |
|---|---:|---:|---:|---:|
| xp_budget | 2116 | 62.2 | +120 | — |
| ridge_h3_1pos_ft | 1996 | 58.7 | +0 | +182 |
| ridge_h3_ft | 1996 | 58.7 | +0 | +182 |
| ridge_h3_nofixture_ft | 1925 | 56.6 | -71 | +111 |
| xp_ft | 1902 | 55.9 | -94 | — |

## Transfer behaviour

- ridge_h3_ft: tx=0.88, holds=19/34, ≥2tx=10, hits=0
- ridge_h3_1pos_ft: tx=0.88, holds=19/34, ≥2tx=10, hits=0
- ridge_h3_nofixture_ft: tx=0.82, holds=22/34, ≥2tx=10, hits=0
- xp_ft: tx=0.94, holds=6/34, ≥2tx=4, hits=2

## Ablations

- Fixture+DGW vs blanks-only: **+71**
- Structural 2tx vs same-pos beam: **+0**
- ridge_h3_ft vs xp_ft: **+94**
- Best FT vs xp_budget (2116): **-120**

## Gate verdict

**PASS — ridge_h3_ft leads and beats stage21 (vs xp +94, vs s21 +182)**

## Plots

- `data/plots/season_climb_stage22.png`

## Output

- `data/processed/season_climb_stage22.csv`
