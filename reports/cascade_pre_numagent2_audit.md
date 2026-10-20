# Cascade work vs Numagent 2.0 — mechanics audit

**Date:** 2026-10-19  
**Trigger:** Numagent 2.0 (`score_xp_v2`) merged to main; confirm everything run
today uses the same assembler.

## Verdict

| Workstream | Branch / PR | Scorer actually used | Aligned to `score_xp_v2`? |
| --- | --- | --- | --- |
| Multistage captain objective | `cursor/multistage-captain-objective-cc3d` | `xp_on_pot` / Opta λ pots | **No** — legacy |
| Multistage FT path | `cursor/multistage-ft-path-cc3d` | `xp_on_pot` | **No** — legacy |
| Minutes overlay | `cursor/multistage-minutes-overlay-cc3d` | scales xmi into `xp_on_pot` path | **No** — legacy |
| Match-band calibration | `cursor/match-band-calibration-cc3d` | MC λ → `xp_on_pot` | **No** — legacy (parked) |
| Player G/A finish | `cursor/player-ga-finish-cc3d` | `compute_xp(ga_mode=…)` | **No** — legacy (parked) |
| Rank tracker | `cursor/rank-tracker-cc3d` | `xp_on_pot` + `compute_xp` | **No** — legacy (parked) |
| Short-horizon scenario | `cursor/short-horizon-scenario-cc3d` (pre-align) | `xp_on_pot` + Shin pots | **No** — superseded |
| Short-horizon scenario | **this branch** `cursor/score-xp-v2-align-cc3d` | `numagent_xp.score_team_side` → `score_xp_v2` | **Yes** |
| Multistage MC+MILP / captain / FT / overlay | **this branch** (rerun) | Opta λ → `cascade_score_v2` → `score_xp_v2` | **Yes** |
| Oracle-rank cascade | **this branch** (rerun) | Opta λ → `score_team_side` | **Yes** (viable) |

Reports under `reports/multistage_*` that say “score_xp v2” while naming
`xp_on_pot` are **mislabelled**: that was the pre-2.0 share×λ formula, not
`protocol.primary_scorer` / `src/markets/numagent_xp.py`.

## What Numagent 2.0 requires

Locked in `experiments/protocol.json` → `numagent_2_0` / `primary_scorer`:

1. AH+OU (+ BTTS live) → Dixon–Coles `M` → `xG` / `xCS` / `xGC`
2. `G_team = α_G × xG`, `A_team = α_A × xG`
3. Same-season-only `rG`/`rA` (RATE_M0=270); share-SE → position×minutes prior
4. Assembler: `score_team_side` / `score_fixture` → column `score_xp_v2`
5. Strategiser picks on that μ (L1 / L3 / L4) — not `forecast_xp.xp_on_pot`

## Alignment done here

`src/models/short_horizon_cascade.py` rewritten on main’s Numagent 2.0 stack:

- Centreline = FD opening fits → `score_team_side` (same as L4 `project_week_score_xp_v2`)
- Scenarios = sample λ ±15%, rebuild `M`, freeze rates
- Path FT = L1 then L3
- Mean-layer vs L4 projector correlation reported as a sanity check

## Long-horizon suite re-run

See `reports/long_horizon_numagent2_rerun.md` (2026-10-19). Multistage +
oracle-rank now use `cascade_score_v2` / `score_team_side`.

## Still parked (do not re-run as live screens)

Until rewired onto `numagent_xp`:

- Match-band / player-finish / rank-tracker likelihood screens that call
  `xp_on_pot` or `compute_xp` as the candidate score

Any new screen must name `score_xp_v2` and import from
`src/markets/numagent_xp.py` (or L4 `project_week_score_xp_v2` /
`cascade_score_v2`).
