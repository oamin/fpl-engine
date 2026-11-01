# Locked — freeze shares under scout m

**Locked:** 2026-11-01 (PI)  
**Protocol:** `score_xp_v2_scout_m_share_policy` · `l4_opta_joint_path_scout_m_share_policy`  
**Lock report:** this file

## What is locked

When scout minutes are applied (rolling-3 as-of, zeros included — **not** one-blank sticky):

1. **Appear / CS / GC** use scout `m`.
2. **`sG` / `sA` freeze** from expanding as-of mean `m` (backroom), projected onto the scout selection board.

Do **not** recompute goal/assist shares from scout `m`. Product expanding-as-of boards (no scout overlay) are unchanged. Sticky one-blank→0 remains parked.

## Evidence

| Screen | Result |
| --- | --- |
| Likelihood freeze − recompute | **promote** Δ log **+0.0064** [+0.0048, +0.0082] |
| Path climb (Layer-1, `value_captain=false`) | inconclusive +0.883 [−1.209, +2.983] — not a park; freeze cuts hits, loses 2024-25 |

Reports: `score_xp_v2_scout_m_share_policy.md`, `l4_opta_joint_path_scout_m_share_policy_gate.md`.

## Call sites

- `src/strategy/layer4_mupost_priced.rebuild_week0_with_scout_share_policy` — default `share_policy="share_freeze"`
- `project_week_score_xp_v2(..., minutes_by_element=scout_m, share_minutes_by_element=expand_m)`
- Horizon: `build_l3_centreline_horizon` / `build_mupost_priced_horizon` with the same pair

## PI

Ollie: lock freeze under scout m (2026-11-01).
