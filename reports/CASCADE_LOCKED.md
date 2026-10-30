# Cascade / L3 — locked stack

**Updated:** 2026-10-30  
**Product forecast lock:** [`L3_CENTRELINE_FORECAST_LOCK.md`](L3_CENTRELINE_FORECAST_LOCK.md) · `protocol.l3_centreline_forecast`  
**Numagent index:** [`LOCKED.md`](LOCKED.md)

Staging lineage: `cursor/club-expanding-gf-ga-2025-26-cc3d` → stacked tips.  
Product landing: PR into `main` from the consolidated tip.

---

## Forecast μ (locked 2026-10-30)

| Horizon | λ source | Board |
| --- | --- | --- |
| Near `t`/`t+1` (priced) | Betfair (live) / FD AH+OU(+BTTS) hist → DC \(M\) moments | `score_team_side` → `score_xp_v2` |
| Far `t+2…` (unpriced) | **Opta** place rankmean, **`venue_mode=blind`**, identity shrinks → λ → \(M\) moments | **same** `score_team_side` |

Strategy horizon board: freeze rates/shares/α at `t`; Opta λ for every `g ∈ [t,end]`  
(`protocol.l3_opta_lambda_horizon`). Product default **H=5** via
`src/strategy/l3_centreline_horizon.py` → live `numagent_live_pick` /
hist `build_l3_centreline_horizon`.

**Generative MC-mean is not μ** (viability B1/B2; horizon snapshots).  
`k∼M` strands remain for scenario/path only.

---

## Engine defaults (locked / cascade_default)

| Piece | Protocol / default | Note |
| --- | --- | --- |
| **L3 centreline forecast** | `l3_centreline_forecast` = `locked` 2026-10-30 | FD near / Opta blind far → `score_team_side` |
| **Strategy horizon board** | `l3_opta_lambda_horizon` = `locked_strategy_horizon` | L3@t + Opta λ |
| **Discrete CS/GC likelihood (eval)** | `l3_discrete_cs_gc_likelihood` = `locked` 2026-10-30 | GKP/DEF Bern/cat from \(M\); μ unchanged |
| **Opta club λ** | `cascade_long_match_outcome` + `cascade_long_blind_venue_lock` | blind; `AWAY_SHRINK=LAM_SHRINK=HOME_ADV=1` (PI 2026-10-29) |
| **cascade_short strands** | `cascade_short_generative` | Centreline \(M→k∼M\); **not** μ; no λ ±15% jitter |
| **cascade_long strands** | `cascade_long_match_outcome` | Opta λ → \(M\|λ→k∼M\); **not** μ; props off |
| **Minutes on gen paths** | `DEFAULT_LINEUP_COV=False` | As-of mean / live xmi; FFScout = pool WHO only |
| **Rate baseline** | `score_xp_v2_rate_baseline_prior` = `cascade_default` | Prior + same-season update |
| **Share / xG pool** | full club `m>0` | No thin-board renorm; \(w=r·m/90\) (π∝r parked) |
| **Joint path transfers** | `cascade_joint_path_plan` = `wired` | Every horizon GW; `e=max(0,k−f)` |
| **Transfer-rule path-V** | `cascade_path_transfer_only` / `xi_ft` | Captain double **off**; bank pot separate |
| **Cascade γ** | `cascade_gamma` = `locked` | `DEFAULT_GAMMA=1.0` |
| **FT bank pot** | `plan.bank_pot` ledger | Constrained; **not** in \(V\) |
| **FT potential energy** | `DEFAULT_FT_PE_NU=0` | Diagnostic only if \(ν>0\) |
| **Bench α** | `cascade_path_bench_weight` = `rejected` | Removed from path objective |

---

## Parked / diagnostic (not product promotes)

- Gen MC-mean as forecast μ — clearly loses (`cascade_viability_batch`, `cascade_short_mean_tt1_vs_gen`)
- Factored five-term vs Gaussian-on-total — park_clear_lose (`m_discrete_event_likelihood_screen`)
- Mode/median \(M\) as μ — parked (Arsenal dig)
- Opta own-λ ×α≠1 — parked; keep α=1
- place×venue / expanding H/A / FD venue-scale / margin-shrink / lag-1 / remaining-balance — parked under blind lock
- Path-V gen vs mean MILP — diagnostic only
- Deadline-causal minutes — `keep_watching` diagnostic
- Player G/A count likelihood — not yet screened
- **Open dig:** player G overprint on cascade / xi_ft (full-share default locked)

---

## Key modules

| Role | Path |
| --- | --- |
| Mean board | `src/markets/numagent_xp.py` |
| FD short fits | `src/models/short_horizon_cascade.py` |
| Opta λ / place | `src/models/opta_gf_ga_horizon.py` |
| Horizon board recipe | `src/eval/l3_opta_lambda_horizon.py` |
| Rate baseline | `src/markets/rate_baseline.py` |
| Gen strands (not μ) | `src/markets/short_horizon_generative.py`, `src/markets/cascade_score_v2.py` |
| Path / pool | `src/models/multistage_mc_milp.py`, `src/models/path_transfer_plan.py`, `src/models/player_pool.py` |
