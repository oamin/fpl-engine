# Stage 12 — Long-horizon prior longevity

Question: do the same priors **rank / track rates over the next H gameweeks**, not only the immediate next match?

For each player-GW t (with ≥3 prior GWs and a full next-8 window):
- predictor = leakage-free roll3 / expanding rate at t
- outcome = **mean** of the target over GWs `[t, t+H)`
- horizons H ∈ {1, 3, 5, 8}

Universe focus: **regulars with xMi ≥ 45** (known-ish minutes).

## Corr vs horizon (regulars xmi≥45)

| scope | pair | H=1 | H=3 | H=5 | H=8 |
|---|---|---:|---:|---:|---:|
| MID+FWD | roll3 xG → goals | 0.191 | 0.285 | 0.338 | 0.374 |
| MID+FWD | exp xG → goals | 0.238 | 0.374 | 0.442 | 0.496 |
| MID+FWD | roll3 goals → goals | 0.128 | 0.196 | 0.224 | 0.253 |
| GKP+DEF | roll3 CS → CS | 0.063 | 0.070 | 0.064 | 0.112 |
| GKP+DEF | exp CS → CS | 0.124 | 0.171 | 0.213 | 0.268 |
| DEF+MID | exp DefCon hit → hit | 0.305 | 0.432 | 0.495 | 0.557 |
| DEF+MID | roll3 DefCon hit → hit | 0.261 | 0.358 | 0.389 | 0.419 |
| ALL | xMi → points | 0.161 | 0.201 | 0.207 | 0.212 |
| ALL | xMi → minutes | 0.387 | 0.391 | 0.368 | 0.355 |
| ALL | roll3 pts → points | 0.128 | 0.177 | 0.185 | 0.203 |
| ALL | exp pts → points | 0.175 | 0.268 | 0.318 | 0.364 |

## Read

- If corr **rises** with H → real rate signal buried in match noise (longevity OK).
- If corr **flat/falls** → prior doesn’t persist; not useful even as a medium-term rate.
- xMi → minutes should stay high; xMi → points may rise slowly via appearance.

## Plots

- `data/plots/long_horizon_corr.png`

## Output

- `data/processed/long_horizon_backtest.csv`
