# Stage 8 — Roll-3 xG / xA / xGI → next-match output

Prior **3-GW rolling mean** of xG, xA, xGI (and naive goal/assist rates) vs realised next-GW events. Leakage-free (`shift` + rolling).

- Raw player-GW rows: **29338**
- Backtest requires ≥ 3 prior GWs

## Headline (scope=ALL)

### All rows (incl. 0 minutes)

| pair | n | MAE | R² | corr | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|
| xG → goals | 26819 | 0.059 | 0.024 | 0.270 | 0.034 | 0.036 |
| xA → assists | 26819 | 0.050 | 0.025 | 0.195 | 0.032 | 0.023 |
| xGI → G+A | 26819 | 0.100 | 0.079 | 0.314 | 0.066 | 0.059 |
| xGI → points | 26819 | 1.120 | -0.177 | 0.389 | 1.152 | 0.059 |
| prior goals → goals | 26819 | 0.057 | -0.167 | 0.199 | 0.034 | 0.034 |
| prior assists → assists | 26819 | 0.056 | -0.208 | 0.150 | 0.032 | 0.032 |
| prior G+A → G+A | 26819 | 0.104 | -0.116 | 0.248 | 0.066 | 0.066 |
| prior points → points | 26819 | 1.043 | 0.160 | 0.491 | 1.152 | 1.161 |
| roll3 mins → mins (ref) | 26819 | 11.420 | 0.601 | 0.787 | 25.159 | 25.280 |

### Minutes ≥ 60 only

| pair | n | MAE | R² | corr | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|
| xG → goals | 7043 | 0.164 | 0.019 | 0.239 | 0.110 | 0.092 |
| xA → assists | 7043 | 0.145 | -0.013 | 0.131 | 0.102 | 0.060 |
| xGI → G+A | 7043 | 0.279 | 0.019 | 0.233 | 0.212 | 0.152 |
| xGI → points | 7043 | 3.682 | -1.339 | 0.110 | 3.817 | 0.152 |
| prior goals → goals | 7043 | 0.165 | -0.156 | 0.168 | 0.110 | 0.092 |
| prior assists → assists | 7043 | 0.161 | -0.202 | 0.105 | 0.102 | 0.084 |
| prior G+A → G+A | 7043 | 0.294 | -0.154 | 0.171 | 0.212 | 0.177 |
| prior points → points | 7043 | 2.735 | -0.394 | 0.055 | 3.817 | 3.107 |
| roll3 mins → mins (ref) | 7043 | 19.447 | -12.986 | 0.291 | 85.418 | 68.015 |

## MID / FWD — xGI → G+A (mins ≥ 60)

| pos | n | MAE | R² | corr |
|---|---:|---:|---:|---:|
| DEF | 2738 | 0.178 | -0.035 | 0.078 |
| MID | 2934 | 0.376 | -0.049 | 0.135 |
| FWD | 687 | 0.540 | -0.088 | 0.115 |

## Plots

- `data/plots/xgi_roll3_backtest.png`
- `data/plots/xgi_roll3_by_position.png`

## Output

- `data/processed/xgi_roll3_backtest.csv`

## Read

- Compare **xG → goals** vs **prior goals → goals** (does xG beat counting?).
- High corr + low R² is common for rare events; decile curves matter more than R².
- If xGI ≉ prior GI for next G+A, prefer the simpler counting rate.
