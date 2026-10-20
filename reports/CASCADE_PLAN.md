# Cascade rebuild plan (post–Numagent 2.0 freeze)

**Branch:** `cursor/cascade-rebuild-plan-cc3d` (build off current `main`).  
**Contract:** [`CASCADE_CONTRACT.md`](CASCADE_CONTRACT.md).  
**Scorer:** `score_xp_v2` only. Surfaces: `cascade_score_v2`, `layer4_mu_inject`, L3/L4.

This replaces the pre-v2 PR forest (#73–#87). Do not reopen those branches onto main.

---

## Architecture

```
decision GW t
├── Short horizon  t : t+1
│   ├── FD (hist) / Odds (live) → Dixon–Coles λ
│   ├── freeze as-of rates (GW < t)
│   ├── score_team_side → μ boards
│   ├── mean: L1 / L3 (current advisor)
│   └── path: μ boards → pools_from_mu_boards → L4  (wired; screen before promote)
└── Long horizon  t+2 … t+n
    ├── Opta ranks → GF/GA (±MAE) → λ → ClubPath.observe
    ├── cascade_score_v2 / score_pool_on_lam_pots
    ├── MILP frequency (squad_sum | xi_plus_captain | xi_plus_captain_ft)
    └── optional hand-off: priced λ observe before Opta roll (plumbing until path-V)
```

---

## What not to replay

Already parked under `score_xp_v2` — treat as closed evidence unless the plan changes:

| Arm | Decision | Report |
| --- | --- | --- |
| Share seeding | `keep_frozen_shares` | `share_seed_screen.md` |
| Assist α_A = team xA/xG | `keep_pooled_alpha_a` | `assist_alpha_screen.md` |
| Match-band z | keep baseline z=1.0 | `match_band_calibration.md` |
| Player finish scales | keep share×λ | `player_ga_finish.md` |
| Rank→GF/GA map | keep OLS baseline | `rank_tracker.md` |

---

## Rebuild sequence (one comparison at a time)

### A. Short horizon — L4 inject path (next code)

1. Keep mean-layer FD cross-check vs `project_week_score_xp_v2` (sanity only).
2. Scenario strands already score via `score_team_side`; attach prices as today.
3. **New:** convert scored t / t+1 boards → `pools_from_mu_boards` → optional `solve_horizon_transfers` (H=2, γ locked). Default advisor stays L1→L3 until a gated path-V / stick-rate screen clears.
4. Helper: `scored_weeks_to_l4_pools` in `short_horizon_cascade.py`.

### B. Long horizon — path objective (deferred screen)

1. Keep Opta λ → `ClubPath` → `cascade_score_v2`.
2. Objectives already on main code path: `squad_sum`, `xi_plus_captain`, `xi_plus_captain_ft`, minutes overlay — rejudge only if the **μ injection / scorer** changed the board (it did not for Opta λ path).
3. Next **new** pre-registered arm: priced→Opta hand-off as a **path-V or player-log** comparison (plumbing exists; no promote yet).

### C. Live glue (later)

`numagent_live_pick` stays Odds → L3 → L4. Cascade does not replace live; it advises horizon risk.

---

## Pre-registered comparisons (queue)

Register in `experiments/protocol.json` before reading totals. Order:

1. **Short L4-inject path V vs L1→L3 path V** (same frozen rates, same λ samples) — likelihood or stick/path-V diagnostic on closed seasons.
2. **Hand-off on vs off** for long-horizon player-GW log (or path-V) — only after (1) does not clearly lose.
3. Captain / chip simulators stay deferred (protocol).

---

## Definition of done for this branch tip

- [x] Plan + contract on main lineage
- [x] `scored_weeks_to_l4_pools` helper + unit test
- [ ] Comparison (1) registered and run (follow-up commit / PR)
- [ ] Context marks Step 4 complete when (1) lands or is explicitly parked
