# Team pots only (no shares)

Same λ / Poisson machinery as stage 11, evaluated at **team×fixture** grain — no player share allocation.
- Team-fixtures (prior ≥ 3): **700**

| pair | n | corr | decile_corr | Δ top−bot | Brier | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| λ_scored → team goals | 700 | 0.142 | 0.827 | 0.386 | — | 1.333 | 1.462 |
| roll3 team xG → team goals | 700 | 0.099 | 0.764 | 0.357 | — | 1.333 | 1.407 |
| λ_scored → team xG (sanity) | 700 | 0.160 | 0.885 | 0.405 | — | 1.412 | 1.462 |
| λ_assist → team assists | 700 | 0.115 | 0.858 | 0.429 | — | 1.243 | 0.931 |
| roll3 team xA → team assists | 700 | 0.081 | 0.468 | 0.306 | — | 1.243 | 0.897 |
| λ_scored → team G+A | 700 | 0.137 | 0.831 | 0.771 | — | 2.576 | 1.462 |
| Poisson P(CS) → team CS | 700 | 0.082 | 0.544 | 0.086 | 0.207 | 0.246 | 0.314 |
| roll3 team CS → team CS | 700 | 0.037 | nan | 0.000 | 0.239 | 0.246 | 0.255 |
| p_not_lose → team CS | 700 | 0.260 | 0.890 | 0.386 | 0.243 | 0.246 | 0.500 |
| λ_conceded → team CS (expect neg) | 700 | -0.104 | -0.605 | -0.086 | 0.651 | 0.246 | 1.336 |
| −λ_conceded → team CS | 700 | 0.104 | 0.605 | 0.086 | 0.246 | 0.246 | -1.336 |

## vs player-level (stage 11, mins≥60)

| grain | goals-ish | CS |
|---|---|---|
| team pot | λ→team goals corr above | Poisson / p_not_lose above |
| player (λ×share / P(CS)) | goals corr 0.19 | Poisson 0.10, p_not_lose 0.28 |

## Plot

- `data/plots/channel_trial_team_pots.png`
