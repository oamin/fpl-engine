# Lock — captain double on joint_path_opta_decay

**Locked:** 2026-10-31 (PI)  
**Protocol:** `joint_path_opta_decay` (objective) · screen `l4_opta_joint_path_captain_climb`  
**Status:** locked

## Product

| Piece | Lock |
| --- | --- |
| Strategiser | `joint_path_opta_decay` (H=12, ρ=0.85→0, wired μ) |
| Objective | `xi_plus_captain` — `value_captain=True` |
| Glue default | `plan_centreline_path(..., value_captain=True)` |
| `ft_pe_nu` | 0 (unchanged; FT PE climb inconclusive) |
| Realised (hist screens) | XI after FPL autosubs + captain/VC − hits |

Transfer-rule isolation screens still pass `value_captain=False` explicitly.

## Gate

| Comparison | Mean Δ/GW | Interval | Verdict |
| --- | ---: | --- | --- |
| `decay_xi_plus_captain` − `decay_xi_ft` | **+3.483** | [+1.866, +5.267] | **promote** |

By season: +7.3 / +0.5 / +4.3 / +1.9 (all ≥ 0).

Report: `reports/l4_opta_joint_path_captain_climb_gate.md`.

## Notes

- Autosubs were 0 on the closed climbs (high as-of-minute XIs rarely blank). VC path is coded.
- Haul / CVaR captain overrides stay parked.
- Parent strategiser lock: `reports/JOINT_PATH_OPTA_DECAY_LOCK.md`.
