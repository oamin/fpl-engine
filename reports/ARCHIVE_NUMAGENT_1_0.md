# Archive — Numagent 1.0 / `score_xp`

**Status:** superseded 2026-10-19. **Product on main is Numagent 2.0** (`protocol.numagent_2_0`, `reports/LOCKED.md`).

This index is the **in-place archive**. Files stay where they are so the as-of audit, the published `score_xp`−`score_exp_points` comparison, and the v2−`score_xp` gate baseline keep working. Do not delete these modules or move them off the import path.

## Roles

| Thing | Role after archive |
| --- | --- |
| Numagent 1.0 | GW6-era live path: XI/captain from captured `ep_next`; `score_xp` shadow |
| `score_xp` | Historical climb replay + **named gate baseline** that lost to `score_xp_v2` |
| Live product | `numagent_live_pick` + `score_xp_v2` + L4 (not anything below) |

Protocol: `numagent_1_0`, `primary_scorer.score_xp_status`, `comparisons` / `comparisons_note`, `decision_layer` (frozen historical batch).

## Code that must stay importable

| Path | Why it stays |
| --- | --- |
| `src/live/scorer.py` | Gates still assert `SCORE_COL == score_xp` and `live_choice` |
| `src/live/deadline.py` | Gates assert `choice=`; GW6-era deadline collector |
| `src/live/wildcard_plan.py` | `numagent_1_0.entrypoints` |
| `src/live/whatif.py` | `numagent_1_0.entrypoints` |
| `src/models/xp_engine.py` | As-of audit + classic `compute_xp` / climbs |
| `src/models/forecast_xp.py` | Opening-pot XP for archived live path |
| `src/markets/build_score_xp_seasons.py` | Rebuilds `data/processed/score_xp_*.csv` |
| `src/eval/honest_pool.py` | Protocol comparison stack on `score_xp` |
| `src/eval/decision_spec.py` | `SCORE_COLUMN = score_xp` for the frozen decision batch |

## Data that must stay

- `data/processed/score_xp_{2022_23,2023_24,2024_25,2025_26}.csv`
- `data/processed/score_xp_closed.csv` (+ summaries)
- `data/processed/procedure_intervals.json`
- Evidence: `reports/score_xp_v2_gate.md`, `reports/procedure_audit.md`

## GW6 / 1.0 reports (path list only — files not moved)

- `reports/live_dry_run_gw6.md`
- `reports/live_deadline_gw6.md`
- `reports/live_whatif_gw6.md`
- `reports/live_minutes_gw6.md`
- `reports/wildcard_plan_gw6_dry_run.md`
- `reports/numagent_3271409_gw6.md` (+ `.json`)
- `reports/chip_calendar_gw6.md`
- `reports/stack_audit_gw6.md`
- `reports/gw6_milp_worth_20261008.md`
- `reports/gw6_predeadline_checklist_20261008.md`
- String-agent GW6 companions under `reports/string_*_gw6*` / `stragent_3271409_gw6.md` (separate agent; not Numagent)

## Stage climb archaeology

See `reports/SUPERSEDED.md` (stages 14–34) and root `CONTEXT.md` (stages 0–48). Park narrative: `.cursor/ideas.md`.

## Do not

- Use 1.0 / `ep_next` / `score_xp` for live picks or new screens (except as the named v2 gate baseline).
- Merge leftover GW6 / string-horizon draft PRs as the product path.
- Relocate the modules above into a folder that breaks imports.
