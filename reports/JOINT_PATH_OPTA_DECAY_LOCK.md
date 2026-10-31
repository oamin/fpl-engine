# Lock — joint path + Opta far decay (strategiser)

**Locked:** 2026-10-31 (PI)  
**Protocol:** `joint_path_opta_decay`  
**Screen:** `l4_opta_joint_path_climb` (full μ)  
**Status:** locked (product strategiser on L4+Opta H=12 boards)

## Product method

| Piece | Lock |
| --- | --- |
| Boards | `build_l3_centreline_horizon(H=12)` — L4/FD priced `t:t+1` + Opta blind far |
| μ | full `mu_post ≡ score_xp_v2` (not component-only) |
| Planner | `pick_path_transfer_plan` every horizon GW (`from_owned`) |
| Candidates | full week-0 pool (no top-48) |
| Objective | `xi_ft` — `value_captain=False`, `ft_pe_nu=0` |
| Opta far | boards `t ≥ near` (near=2): μ and xp_* × **ρ^(t−near+1)** → 0; **ρ=0.85** |
| Realised | XI total_points − hits (no captain double); no chips |

Glue: `src/strategy/centreline_path_plan.py` (`decay=True`).

## Why this arm (PI)

Gated climb on four closed seasons (GW8–38):

| Comparison | Mean Δ/GW | Interval | Gate |
| --- | ---: | --- | --- |
| `joint_path` − `l4_opta_h12` | +1.925 | [+0.08, +3.69] | **promote** |
| **`joint_path_decay` − `l4_opta_h12`** | **+1.783** | [−0.15, +3.88] | inconclusive |
| `joint_path_decay` − `joint_path` | −0.142 | [−1.61, +1.36] | inconclusive |

PI locks **decay** over undecayed path: decay is **positive vs L4 in every season**
(+1.33 / +1.73 / +1.03 / +3.03). Undecayed path loses **2024-25** (−1.07/GW)
despite the formal promote interval. Season consistency preferred for the
strategiser default.

Report: `reports/l4_opta_joint_path_climb_gate.md`.

## Not locked here

| Item | Note |
| --- | --- |
| Undecayed `joint_path` | Formal promote; not product default |
| Component-μ climbs | Separate screen (`l4_opta_joint_path_component_climb`); PI reviewing |
| Live entry | Still L4 oneshot / priced H=2 while Opta far is not live-ready; when H=12 Opta far is used, use this planner + decay |
| μ / scorer | Unchanged — `l3_centreline_forecast` / `score_xp_v2` |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_climb_gate
```
