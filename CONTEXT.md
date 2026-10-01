# FPL Engine — Architecture (v1 front-end)

Predict Fantasy Premier League points via **team pot × player share**, then
baseline/residual. Squad/macro layers come after stages 0–5.

```
ingest (FPL + football-data odds)
  → team×pos pots + channel shares
  → team pots vs 1X2 / OU / AH  (go/no-go)
  → player baseline + residual z
  → assemble μ from market terms that passed
```

## Stages

| Stage | Module | Report |
| --- | --- | --- |
| 0 | Clean slate | `reports/stage_0_clean.md` |
| 1 | `src.ingest` | `reports/stage_1_ingest.md` |
| 2 | `src.features.shares` | `reports/stage_2_shares.md` |
| 3 | `src.models.team_market` | `reports/stage_3_team_market.md` |
| 4 | `src.features.baseline` | `reports/stage_4_baseline.md` |
| 5 | `src.models.player_mu` | `reports/stage_5_player_mu.md` |
| 6 | `src.models.shin_only_mu` | `reports/stage_6_shin_only_mu.md` |
| 7 | `src.models.xmi_historic` | `reports/stage_7_xmi_historic.md` |
| 8 | `src.models.xgi_roll3` | `reports/stage_8_xgi_roll3.md` |
| 9 | `src.models.cs_shin` | `reports/stage_9_cs_shin.md` |
| 10 | `src.models.xmi_points_gate` | `reports/stage_10_xmi_points_gate.md` |
| 11 | `src.models.channel_trial` | `reports/stage_11_channel_trial.md` |
| 12 | `src.models.long_horizon` | `reports/stage_12_long_horizon.md` |
| 13 | `src.models.xp_engine` | `reports/stage_13_xp_engine.md` |
| 14 | `src.models.season_climb` | `reports/stage_14_season_climb.md` |
| 15 | `src.models.season_climb_ml` | `reports/stage_15_season_climb_ml.md` |
| 16 | `src.models.ridge_multiseason` | `reports/stage_16_ridge_multiseason.md` |
| 17 | `src.models.ridge_starters` | `reports/stage_17_ridge_starters.md` |
| 18 | `src.models.season_climb_budget` | `reports/stage_18_season_climb_budget.md` |
| 24 | `src.models.xp_v2_lgbm` | `reports/stage_24_xp_v2_lgbm.md` |
| 25 | `src.models.stage_25_xp_saves_blend` | `reports/stage_25_xp_saves_blend.md` |
| 26 | `src.models.stage_26_fwd_team_fade` | `reports/stage_26_fwd_team_fade.md` |
| 27 | `src.models.stage_27_play_audit` | `reports/stage_27_play_audit.md` |

```bash
uv run python -m src.run_frontend
PYTHONPATH=. uv run --no-sync python -m src.models.shin_only_mu
PYTHONPATH=. uv run --no-sync python -m src.models.xmi_historic
PYTHONPATH=. uv run --no-sync python -m src.models.xgi_roll3
PYTHONPATH=. uv run --no-sync python -m src.models.cs_shin
PYTHONPATH=. uv run --no-sync python -m src.models.xmi_points_gate
PYTHONPATH=. uv run --no-sync python -m src.models.channel_trial
```

Stage 6 rebuilds μ using **Shin P(win)/P(lose) only** (no OU/AH): team pot tilt → share × tilt → player μ.  
Stage 7 builds historic **xMi** = **3-GW rolling mean minutes** (backtested).  
Stage 8–12: channel diagnostics (next-GW / long-horizon); informative but not the model gate.  
Stage 13: thin **xP engine** (xMi × market λ/CS × shares × DefCon).  
Stage 14: **stripped season climb** — primary gate for predictive models.

## Model evaluation (locked)

**Primary gate:** `src.models.season_climb` — each GW pick a position-legal XI by score
(no budget / chips / transfers); captain = top score in XI; compare **cumulative
actual points** vs baselines (`exp_points`, `xmi`, `price`, `random`).

Pass bar: finish clearly above `exp_points` over the season (same captain rule).

Pooled Spearman / next-GW R² are diagnostics only — they can disagree with XI-tail
ranking (stage 13 IC failed; stage 14 climb passed for xP).

Stage 15: walk-forward **ML scorers** (`ml_ridge` / `ml_hgb` / `ml_lgbm` / `ml_resid`)
on the same climb gate — train on `gw < t` only.

Stage 16: **multi-season walk-forward Ridge** (2022/23–2025/26 train history;
eval 2025/26 climb). Prefer `ridge_ms` when it beats single-season Ridge + xP.

Stage 17: **starter-only / no-xMi / per-position Ridge** on eligible 60′ pool
(fresh seasons 23/24–25/26 + recency weights). `ridge_global_starters` beat
`ridge_pos` slightly; both beat xP/exp on the eligible climb.

Stage 18: **budgeted climb** (£100.0m / 15 → XI). Shows premium inflation in
free climb; under budget ridge ≈ xP and still beats exp_points.

## Odds API

Free tier is tiny. Historical path uses **football-data.co.uk** only (no Odds API).
Live meso later may use `data/cache/odds_snapshot.json`. See `.cursor/rules/odds-api-quota.mdc`.

## Keep / wipe

Kept: `.env.example`, `.gitignore`, odds quota rule, `data/cache/odds_snapshot.json`.
Wiped: prior analyse/λ/DefCon/meso matrix code and derived CSVs.
