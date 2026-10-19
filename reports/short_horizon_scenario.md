# Short-horizon scenario layer (priced t : t+1) — Numagent 2.0

Advisor layer on market-priced weeks. Scorer: **`score_xp_v2`** (`src/markets/numagent_xp.py`). Does **not** replace the published L4 mean path. Rates frozen at the deadline; λ sampled inside ±15% of FD Dixon–Coles fits; path FT via L1 → L3.

## Decision

**keep_scenario_advisor**

- Keep advisor if path stick ≥ independent t∩t+1 overlap + 0.05. Published `score_xp_v2` stays the centreline mean layer either way.
- path_signal=True · mean_layer_quiet=True (|ΔxP|<0.25)

## Summary

| season | 2024-25 |
| scorer | score_xp_v2 |
| deadlines ok | 2 / 2 |
| mean ΔV (path − indep) | -4.904 |
| mean stick (path) | 0.933 |
| mean indep overlap t∩t+1 | 0.333 |
| mean hits t+1 | 0.00 |
| mean |ΔxP| vs centreline (t) | 0.0936 |
| mean-layer vs L4 projector corr | 1.000 |

## Per deadline

| GW | V_indep | mean V_path | ΔV | stick | hits | |ΔxP| | L4 corr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 124.52 | 121.34 | -3.17 | 0.93 | 0.00 | 0.0961 | 1.000 |
| 20 | 127.84 | 121.20 | -6.63 | 0.93 | 0.00 | 0.0912 | 1.000 |

## Mechanics (aligned to Numagent 2.0)

1. Scoreline: FD opening AH+OU → Dixon–Coles `M` (`scoreline.build_matrix`).
2. Team pots: `α_G/α_A × xG_M`; defence from `xCS` / `xGC` on the same `M`.
3. Player rates: same-season GW `< t` only (`asof_share_history` + RATE_M0).
4. Assembler: `numagent_xp.score_team_side` → `score_xp_v2` (share-SE shrink on).
5. Scenario: sample λ ±15%; rebuild `M`; freeze rates; do not replace mean μ.
6. Path: L1 free rebuild at t, L3 FT/hits at t+1.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.models.short_horizon_cascade
```
