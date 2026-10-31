# `score_xp_v2` vs `score_xp` — closed seasons (prep)

Candidate score column: **`score_xp_v2`** (method spec was Numagent 2.0). Not gated. Same pool as the 2025/26 pilot (minutes > 60, σ_m = 0, FD openings). Cluster interval uses the protocol bootstrap (resample GWs within each season).

## Cluster interval (GW 5–38)

- Mean Δlog-score (score_xp_v2 − score_xp): **+0.0773** [+0.0711, +0.0841]
- Gameweeks per season: {'2022-23': 33, '2023-24': 34, '2024-25': 34, '2025-26': 34}

## By season

- **2022-23**: n=6864, mean Δ=+0.1003, MAE v2/sxp=2.278/2.272
- **2023-24**: n=6956, mean Δ=+0.0694, MAE v2/sxp=2.314/2.294
- **2024-25**: n=6892, mean Δ=+0.0535, MAE v2/sxp=2.264/2.239
- **2025-26**: n=6856, mean Δ=+0.0865, MAE v2/sxp=2.345/2.375

## Gate checklist (remaining)

- [ ] As-of audit on the feature path used here
- [ ] Pre-register `score_xp_v2` vs `score_xp` without dropping the required `score_xp` vs `score_exp_points` interval
- [ ] `write_gated_report`

## Artefacts

- `data/processed/score_xp_v2_closed.csv`
- `data/processed/score_xp_v2_closed_gw.csv`
- `data/processed/score_xp_v2_closed_summary.json`
- `data/plots/score_xp_v2_closed_dlog.png`

Older `numagent_xp_closed*` filenames are superseded by these.
