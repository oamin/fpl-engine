# Stage 20 — Hybrid μ FT climb (dual-layer)

Gemini recommendation: keep Ridge for free rebuild / XI; dampen swaps with xP.

- **hybrid_ft (dual):** init+XI = Ridge; transfers = `0.6·z(xP) + 0.4·z(Ridge)`
- **hybrid_all_ft:** hybrid for init + transfers + XI
- **ridge_ft / xp_ft:** single-μ controls (same V as stage 19b)

- V: Σ γ^h XI_score (H=3, γ=0.9) − 4·hits; hold unless ΔV ≥ 1.25
- FTs: 1/GW stack to 5; max hits explored 2
- GWs: **5–38** (n=34)

## Final standings (captain ×2 − hits)

| method | mode | total | mean/GW | vs hybrid_ft |
|---|---|---:|---:|---:|
| xp_budget | budget | 2116 | 62.2 | +401 |
| ridge_global_starters_budget | budget | 2114 | 62.2 | +399 |
| xp_ft | ft | 1848 | 54.4 | +133 |
| hybrid_all_ft | ft | 1825 | 53.7 | +110 |
| ridge_global_starters_ft | ft | 1746 | 51.4 | +31 |
| hybrid_ft | ft | 1715 | 50.4 | +0 |

## Transfer behaviour

- hybrid_ft: mean_tx=1.00, holds=5/34, max_ft=2, hits=2
- hybrid_all_ft: mean_tx=1.00, holds=3/34, max_ft=2, hits=2
- xp_ft: mean_tx=0.94, holds=2/34, max_ft=1, hits=0
- ridge_ft: mean_tx=0.97, holds=4/34, max_ft=2, hits=1

## Deltas

- hybrid_ft vs xp_ft: **-133**
- hybrid_ft vs ridge_ft: **-31**
- hybrid_ft vs hybrid_all: **-110**
- hybrid_ft vs ridge_budget: **-399**

## Gate verdict

**FAIL — hybrid_ft does not beat ridge_ft (-31); vs xp -133**

## Plots

- `data/plots/season_climb_hybrid.png`

## Output

- `data/processed/season_climb_hybrid.csv`

## Read

- Dual-layer tests whether Ridge μ + smooth swap signal closes the FT gap.
- hybrid_all isolates whether dual-layer (vs blend everywhere) matters.
- Budget ceiling remains an upper bound, not the fair peer.
