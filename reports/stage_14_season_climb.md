# Stage 14 — Stripped season climb

Each GW pick a **position-legal XI** by score from the full pool, bank actual points (captain = top score in XI, ×2).

**Stripped out:** budget, chips, transfer continuity, bench.
**Kept:** 1 GKP + legal formation (best of common shapes by score sum).

Priors are leakage-free at GW t (`n_prior ≥ 3`).

## Final standings (captain ×2)

| method | total | mean/GW | vs exp_points |
|---|---:|---:|---:|
| xp | 2137 | 61.1 | +215 |
| blend_xp_exp | 2105 | 60.1 | +183 |
| price | 1997 | 57.1 | +75 |
| exp_points | 1922 | 54.9 | +0 |
| roll3_points | 1797 | 51.3 | -125 |
| xmi | 1619 | 46.3 | -303 |
| random | 1293 | 36.9 | -629 |

## vs exp_points by GW (Δ XI pts with captain)

| method | GWs ahead | GWs behind | mean Δ/GW |
|---|---:|---:|---:|
| xp | 23 | 11 | +6.14 |
| blend_xp_exp | 23 | 10 | +5.23 |
| xmi | 9 | 26 | -8.66 |
| price | 20 | 15 | +2.14 |
| random | 7 | 28 | -17.97 |

## Gate verdict

**PASS — blend beats exp_points by +183**

Pass bar: finish ≥ **+10** cumulative pts vs exp_points over the season (captain rule identical across methods).

## Plots

- `data/plots/season_climb.png`

## Output

- `data/processed/season_climb.csv`

## Read

- This tests **ranking quality under XI constraints**, not transfer skill.
- If xP / blend cannot beat exp_points here, MILP will not save it.
