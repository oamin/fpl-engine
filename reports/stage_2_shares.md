# Stage 2 — Team pots + channel shares

- Player rows with shares: **11498**
- Team×pos pot rows: **3028**
- Mean max points-share within pot: **0.622**
- Mean minutes-share (≥60′): **0.343**

## Mean pot points by position

- DEF: 15.84
- FWD: 5.66
- GKP: 3.37
- MID: 20.49

## Outputs

- `data/processed/player_shares.csv`
- `data/processed/team_pos_pots.csv`

Channels: minutes, goals, assists, xG, xA, total_points.
Prior shares = expanding mean of previous appearances (no leakage).
