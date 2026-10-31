# Locked — rate baseline + full-club shares on published / centreline μ

**Locked:** 2026-10-31 (PI)  
**Protocol:** `score_xp_v2_rate_baseline_prior` · `score_xp_v2_full_share_wire`  
**Climb sanity:** `l4_opta_joint_path_mu_wire_climb`

## What is locked

Published and centreline mean boards use:

1. **Historical rate baseline** for attack `rG`/`rA` — prior closed seasons (player else positional) + same-season GW `< t` update (`src/markets/rate_baseline.py`).
2. **Full-club `m>0` shares** — `share_players` from the wider club roster projected onto the selection board (`project_week_score_xp_v2`, Opta centreline).

Soft priors / minutes stay same-season. Locked strategiser remains `joint_path_opta_decay` (ρ=0.85→0).

## Evidence

| Screen | Result |
| --- | --- |
| Rate baseline likelihood | **promote** Δ log **+0.0024** [+0.0015, +0.0035] |
| Full-club vs thin shares likelihood | **promote** Δ log **+0.0101** [+0.0087, +0.0116] |
| Climb: wired decay − legacy | inconclusive −0.250 [−2.167, +1.725] — not a park |
| Climb: wired decay − wired L4 | inconclusive +1.125 [−0.667, +2.951] — still leans ahead |

Reports: `rate_baseline_share_pool_likelihood_gate.md`, `l4_opta_joint_path_mu_wire_climb_gate.md`.

## Call sites

- `src/markets/numagent_xp_pilot.score_gameweek` (`historical_rate_baseline=True`)
- `src/strategy/layer4_mupost_priced.project_week_score_xp_v2`
- `src/strategy/l3_centreline_horizon.project_week_opta_centreline`

## PI

Ollie: keep the wired μ (2026-10-31).
