# Long-horizon audits — Numagent 2.0 rerun

**Date:** 2026-10-19  
**Scorer:** `score_xp_v2` (`src/markets/numagent_xp.py`)  
**Assembler path:** Opta / path λ → Dixon–Coles `M` → `score_team_side`  
**Helper:** `src/markets/cascade_score_v2.py`

Pre-merge cascade arms used `xp_on_pot`. This batch rewires the long-horizon
suite onto the locked Numagent 2.0 assembler and re-runs the diagnostics.

## Suite

| Audit | Config | N | Result artifact |
| --- | --- | ---: | --- |
| Multistage MC+MILP | `squad_sum` + minutes overlay | 12 | `reports/multistage_mc_milp_overlay_trial.md` |
| Multistage captain | `xi_plus_captain` + overlay | 12 | `reports/multistage_xi_captain_overlay_trial.md` |
| Multistage FT path | `xi_plus_captain_ft` + overlay | 12 | `reports/multistage_xi_captain_ft_overlay_trial.md` |
| Multistage FT path | `xi_plus_captain_ft` no overlay | 12 | `reports/multistage_xi_captain_ft_trial.md` |
| Oracle-rank cascade | 2025/26 ceiling | — | `reports/oracle_rank_cascade_2025_26.md` |

## Mechanics change

| Piece | Legacy (pre-merge) | Numagent 2.0 (this run) |
| --- | --- | --- |
| Match scenario | Opta rank→GF/GA → λ (±bands) | **unchanged** |
| Player rates | `add_player_priors` share_xG/xA | same-season `rG`/`rA` (RATE_M0) |
| Minutes | deadline `xmi` | rolling 3-GW mean → `m` |
| Scoreline → xP | Shin-style pot × `xp_on_pot` | `build_matrix(λ)` → moments → `score_team_side` |
| α_G / α_A | implicit in pot | locked pooled α from `fpl_score_rates` |
| MILP | selectable pool on `mc_score` | **same objectives** (squad / XI+C / FT) |

## Headline reads

### Oracle-rank cascade (2025/26)

- **Viable vs flat / prior-rank:** PASS (match MSE + player log)
- Player-GW mean log (`score_xp_v2`, m>60, n_prior≥3):
  - oracle **−2.5538** · prior **−2.5672** · flat **−2.5748**
  - market FD `score_xp_v2` **−2.5725**
- Oracle − market log: **+0.0187** (reference only)

Oracle ranks still clear the cascade ceiling under Numagent 2.0 player scoring.

### Multistage frequency (decision GW6 → GW8–19, seed 7)

Top-frequency names under `score_xp_v2` (overlay on unless noted):

| Objective | #1 | #2 | #3 | #4 | #5 |
| --- | --- | --- | --- | --- | --- |
| squad_sum | Tarkowski | Hall | Groß | Haaland | Lewis-Potter |
| xi_plus_captain | Tzolakis | Tarkowski | Groß | Haaland | Hall |
| FT + overlay | Tarkowski | Groß | Haaland | Hall | Tzolakis |
| FT no overlay | Tarkowski | Groß | Tzolakis | Hall | Lewis-Potter |

Haaland remains a high-frequency captain candidate under `xi_plus_captain`
(freq_captain ≈ 0.24). Overlay vs no-overlay FT boards stay close — overlay
still useful plumbing, not a likelihood claim.

## Still on legacy scorer (not re-run here)

Calibration screens that still call `xp_on_pot` / `compute_xp`:

- `src/eval/match_band_calibration.py`
- `src/eval/rank_tracker.py`
- `src/eval/player_ga_screen.py`

Parked until rewired onto `cascade_score_v2` / `numagent_xp`.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 12 --objective squad_sum
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 12 --objective xi_plus_captain
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 12 --objective xi_plus_captain_ft --minutes-overlay
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 12 --objective xi_plus_captain_ft --no-minutes-overlay \
  --out data/predictions/2026-27/multistage_ft_no_overlay
PYTHONPATH=. python3 -m src.eval.oracle_rank_cascade
```
