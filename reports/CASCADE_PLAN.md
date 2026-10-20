# Cascade rebuild plan (post–Numagent 2.0 freeze)

**Product scorer first:** confirm [`SCORE_XP_V2_CHECKLIST.md`](SCORE_XP_V2_CHECKLIST.md) before new path-V screens.  
**Contract:** [`CASCADE_CONTRACT.md`](CASCADE_CONTRACT.md).

Two branches, **same principle** (`score_xp_v2` / `score_team_side`), **different match inputs**:

| Branch | Module | Horizon | Match source | Buy universe |
| --- | --- | --- | --- | --- |
| **cascade_short** | `src/models/cascade_short.py` → `short_horizon_cascade` | t : t+1 | Live book (live) / FD AH+OU (hist); scenario = generative `k∼M`→lineup→dice | **player_pool** |
| **cascade_long** | `src/models/cascade_long.py` → `multistage_mc_milp` | t+2 … t+n | Opta ranks → GF/GA → λ → ClubPath | **player_pool** |

Shared **player_pool**: FFScout firm ∩ expanding minutes > 60 — `src/models/player_pool.py` (snapshot under `data/predictions/2026-27/selectable_pool/`). Sticky for the season; both branches load the same ids.

---

## Architecture

```
decision GW t
├── player_pool (FFScout ∩ minutes) ── sticky season ids
├── cascade_short  t : t+1
│   ├── live book / hist FD → Dixon–Coles λ (± band)
│   ├── freeze as-of rates (GW < t)
│   ├── mean: score_team_side → μ boards (pool-filtered)
│   ├── scenario: k∼M → lineup cov → attack dice; CS/GC from rolled GA (firm XI later)
│   ├── mean: L1 / L3
│   └── path: scored_weeks_to_l4_pools → L4 (screen before promote)
└── cascade_long  t+2 … t+n
    ├── Opta ranks → GF/GA (±MAE) → λ → ClubPath.observe
    ├── cascade_score_v2 on pool
    ├── MILP frequency (squad_sum | xi_plus_captain | xi_plus_captain_ft)
    └── optional priced→Opta hand-off (plumbing until path-V)
```

---

## What not to replay

| Arm | Decision |
| --- | --- |
| Share seeding | `keep_frozen_shares` |
| Assist α_A = team xA/xG | `keep_pooled_alpha_a` |
| Match-band z / finish / rank map | keep baselines |

---

## Sequence

1. **PI:** tick `SCORE_XP_V2_CHECKLIST.md` — no further scorer screens required.
2. **player_pool** shared by short + long (this tip).
3. Register short L4-inject path V vs L1→L3 only after (1).
4. Hand-off likelihood arm only after (3) does not clearly lose.
