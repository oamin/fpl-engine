# Stage 17 — Starter-only, no-minutes, per-position Ridge

Highest-leverage follow-ups (no captain model yet):

- **No minutes features** (dropped xMi / xp_appear / play-scaled xP terms)
- **Train** only on player-GWs with **minutes ≥ 60**
- **Climb pool** = players with prior-3 max minutes ≥ 60 (leakage-free)
- **Per-position Ridge** (GKP/DEF/MID/FWD)
- **Fresh data:** seasons 2023/24–2025/26 with recency sample weights `{'2023-24': 1.0, '2024-25': 2.0, '2025-26': 3.0}` (no 2022/23)

- Eval GWs: **5–38** (n=34)
- Rows by season: 2023-24=9765, 2024-25=9968, 2025-26=9965

## Final standings (eligible pool, captain ×2)

| method | total | mean/GW | vs exp | vs xP |
|---|---:|---:|---:|---:|
| ridge_global_starters | 2225 | 65.4 | +376 | +122 |
| ridge_pos | 2175 | 64.0 | +326 | +72 |
| xp | 2103 | 61.9 | +254 | +0 |
| blend_xp_exp | 2058 | 60.5 | +209 | -45 |
| price | 1963 | 57.7 | +114 | -140 |
| exp_points | 1849 | 54.4 | +0 | -254 |
| random | 1327 | 39.0 | -522 | -776 |

## Gate verdict

**PASS — ridge_global_starters leads (vs xP +122, vs exp +376; pos vs global -50)**

## Plots

- `data/plots/season_climb_ridge_v2.png`

## Output

- `data/processed/season_climb_ridge_v2.csv`
