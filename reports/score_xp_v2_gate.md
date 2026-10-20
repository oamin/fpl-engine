Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# `score_xp_v2` vs `score_xp` — gated closed-season screen

## Gates

- Engine as-of audit: **passed** (published `score_xp` path).
- `score_xp_v2` as-of: **passed** — rates/priors from GW `< t` only; **minutes on the scored week are an oracle** (historical stack screen; not live `xmi` tags).
- Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

## Pre-registration

- Screen column: `score_xp_v2` (protocol.`score_xp_v2_screen`).
- Pool: minutes > 60 with FD opening AH+OU fit; join to closed `score_xp` (n_prior ≥ 3).
- Minutes: **actual** minutes, σ_m = 0. This screen asks how well the non-minutes stack predicts points, not whether we can tag minutes.
- Odds: football-data opening AH + OU 2.5 (BTTS weight 0).
- Metric: Gaussian log score, σ = 3, on the point mean. `score_xp_v2_se` is stability only.
- Protocol comparison `score_xp` vs `score_exp_points` is unchanged and is carried in this report so `write_gated_report` can run.

## Primary — `score_xp_v2` − `score_xp`

- Mean GW Δlog-score: **+0.0531** [+0.0453, +0.0609].
- Gameweeks: {'2022-23': 33, '2023-24': 34, '2024-25': 34, '2025-26': 34}.
- Player-fixtures: **27568**.

The interval stays **above zero**.

### By season

- **2022-23**: mean Δ=+0.0768, n=6864, MAE v2/sxp=2.285/2.272
- **2023-24**: mean Δ=+0.0435, n=6956, MAE v2/sxp=2.320/2.294
- **2024-25**: mean Δ=+0.0306, n=6892, MAE v2/sxp=2.254/2.239
- **2025-26**: mean Δ=+0.0622, n=6856, MAE v2/sxp=2.364/2.375

## Carried protocol comparison — `score_xp` − `score_exp_points`

- From `procedure_intervals.json`: **+0.0153** [+0.0130, +0.0175].
- Unchanged. Not re-estimated in this screen.

## Artefacts

- `data/processed/score_xp_v2_closed.csv`
- `data/processed/score_xp_v2_closed_gw.csv`
- `data/processed/score_xp_v2_gate_intervals.json`
- `data/plots/score_xp_v2_closed_dlog.png`

**Promotion (2026-10-19):** `score_xp_v2` is the engine primary scorer (`protocol.primary_scorer`). `score_xp` is superseded — this gate is the evidence. Historical climb modules that still read `score_xp` are replay-only.
