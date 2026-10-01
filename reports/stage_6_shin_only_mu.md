# Stage 6 — Shin P(win)/P(lose) only → player μ

Market features: **p_win / p_lose only** (no OU, no AH).

| pos | pot feature |
|---|---|
| GKP | `p_lose` |
| DEF | `p_lose` |
| MID | `p_win` |
| FWD | none (baseline only) |

## Team pot fits

| pos | feature | n | β | R² | decile_r |
|---|---|---:|---:|---:|---:|
| GKP | p_lose | 760 | -2.57 | 0.031 | -0.914 |
| DEF | p_lose | 760 | -15.20 | 0.065 | -0.961 |
| MID | p_win | 760 | 17.91 | 0.149 | 0.980 |
| FWD | none | 748 | 0.00 | 0.000 | nan |

## Player μ vs baseline-only

μ = baseline + residual~(share×pot_tilt, pot_tilt).

| pos | n | R²(resid) | corr_μ | corr_base | Δcorr | lift_μ | lift_base | Δlift | beats? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| GKP | 648 | 0.092 | 0.140 | 0.042 | 0.099 | 1.45 | 0.45 | 1.00 | Y |
| DEF | 2698 | 0.059 | 0.162 | 0.100 | 0.061 | 1.79 | 1.00 | 0.79 | Y |
| MID | 2936 | 0.019 | 0.130 | 0.105 | 0.025 | 1.09 | 0.75 | 0.35 | Y |
| FWD | 691 | 0.000 | 0.166 | 0.166 | 0.000 | 1.66 | 1.66 | 0.00 | N |

## Gate

- Ship DEF/MID fixture term if Δcorr > 0 or Δlift > 0.
- FWD stays baseline-only by design.

## Plots

- `data/plots/stage_6_shin_pot.png`
- `data/plots/stage_6_shin_mu.png`

## Outputs

- `data/processed/team_pot_shin.csv`
- `data/processed/player_mu_shin.csv`
