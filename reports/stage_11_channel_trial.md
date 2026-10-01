# Stage 11 — Trial: team λ × share / Poisson CS / DefCon

Conditioned on **minutes ≥ 60**. Leakage-free team roll3 xG/xGC → opponent-adjusted λ; player share of team attack; Poisson P(CS); expanding DefCon threshold hit rate.

## Goals / assists (MID+FWD)

| pair | n | corr | decile_corr | Δ top−bot | MAE | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| λ×share_xG → goals | 3627 | 0.190 | 0.929 | 0.289 | 0.268 | 0.182 | 0.164 |
| roll3 xG → goals (base) | 3627 | 0.196 | 0.959 | 0.311 | 0.266 | 0.182 | 0.158 |
| roll3 goals → goals (base) | 3627 | 0.139 | nan | 0.296 | 0.273 | 0.182 | 0.167 |
| λ×share_xA → assists | 3627 | 0.111 | 0.881 | 0.154 | 0.206 | 0.144 | 0.095 |
| roll3 xA → assists (base) | 3627 | 0.106 | 0.872 | 0.119 | 0.204 | 0.144 | 0.091 |
| λ×share → G+A | 3627 | 0.164 | 0.971 | 0.325 | 0.415 | 0.326 | 0.259 |
| roll3 xG → G+A (loose base) | 3627 | 0.156 | 0.944 | 0.337 | 0.377 | 0.326 | 0.158 |
| roll3 goals → G+A (base) | 3627 | 0.100 | nan | 0.312 | 0.386 | 0.326 | 0.167 |

## Clean sheets (GKP+DEF)

| pair | n | corr | decile_corr | Brier | Δ top−bot | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poisson P(CS) → CS | 3346 | 0.096 | 0.612 | 0.210 | 0.092 | 0.261 | 0.313 |
| p_not_lose → CS (base) | 3346 | 0.275 | 0.891 | 0.238 | 0.424 | 0.261 | 0.497 |
| roll3 player CS → CS (base) | 3346 | 0.016 | nan | 0.250 | 0.003 | 0.261 | 0.238 |
| roll3 team CS → CS (base) | 3346 | 0.037 | nan | 0.246 | 0.017 | 0.261 | 0.255 |

## DefCon hit (DEF+MID)

| pair | n | corr | decile_corr | Brier | Δ top−bot | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| prior P(DefCon hit) → hit | 5634 | 0.323 | 0.966 | 0.162 | 0.367 | 0.226 | 0.177 |
| roll3 DefCon hit → hit (base) | 5634 | 0.272 | nan | 0.187 | 0.279 | 0.226 | 0.192 |

### By position (headline predictors)

| scope | pair | corr | decile_corr | Δ / Brier |
|---|---|---:|---:|---:|
| MID | λ×share_xG → goals | 0.150 | 0.957 | Δ=0.235 |
| FWD | λ×share_xG → goals | 0.114 | 0.964 | Δ=0.236 |
| GKP | Poisson P(CS) → CS | 0.087 | 0.575 | Brier=0.209 |
| DEF | Poisson P(CS) → CS | 0.098 | 0.633 | Brier=0.210 |
| DEF | prior P(DefCon hit) → hit | 0.285 | 0.929 | Brier=0.190 |
| MID | prior P(DefCon hit) → hit | 0.338 | 0.977 | Brier=0.137 |

## Plots

- `data/plots/channel_trial_ga.png`
- `data/plots/channel_trial_cs_defcon.png`

## Output

- `data/processed/channel_trial.csv`

## Read

- Win = trial corr/decile/Brier clearly beats roll3 / p_not_lose baselines.
- Modest lift still useful for EV ranking among known starters.
