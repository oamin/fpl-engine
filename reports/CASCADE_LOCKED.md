# Cascade — locked stack (consolidation 2026-10-23)

Staging branch: `cursor/club-expanding-gf-ga-2025-26-cc3d` (fast-forward of path-bench tip #101–#106).  
Product landing: PR into `main` from that tip.

## Engine defaults (locked / cascade_default)

| Piece | Protocol / default | Note |
| --- | --- | --- |
| **cascade_short** | `cascade_short_generative` | Centreline `M→k∼M`; no λ ±15% jitter |
| **cascade_long** | `cascade_long_match_outcome` | **PI-approved 2026-10-24**: rankmean±SD latents (`centre_blend=0.5`); assessed expanding ḡ; hierarchical λ→`M\|λ`→`k∼M`; `HOME_ADV=1.0`; no second λ jitter; OLS retired; props off |
| **Minutes on gen paths** | `DEFAULT_LINEUP_COV=False` | As-of mean / live xmi; FFScout = pool WHO only |
| **Rate baseline** | `score_xp_v2_rate_baseline_prior` = `cascade_default` | Prior + same-season update; full `mu_post` gate deferred |
| **Joint path transfers** | `cascade_joint_path_plan` = `wired` | Every horizon GW; `e=max(0,k−f)` |
| **Transfer-rule path-V** | `cascade_path_transfer_only` / `xi_ft` | Captain double **off**; bank pot separate |
| **Cascade γ** | `cascade_gamma` = `locked` | `DEFAULT_GAMMA=1.0` (climb 0.9 stays on legacy climb only) |
| **FT bank pot** | `plan.bank_pot` ledger | `ft_before`/`ft_after` constrained; **not** in \(V\) |
| **FT potential energy** | `DEFAULT_FT_PE_NU=0` | PE optional diagnostic only (`ν>0` explicit) |
| **Bench α** | `cascade_path_bench_weight` = `rejected` | Removed from path objective |

## Parked / diagnostic (not product promotes)

- Gen MC-mean as forecast — clearly loses (`cascade_viability_batch`, `cascade_short_mean_tt1_vs_gen`)
- Path-V gen vs mean MILP — diagnostic only
- Deadline-causal minutes — `keep_watching` diagnostic
- Share-conditional `E[G/A|k]` — candidate gen path-V default; not yet on this tip (#108)
- Hist MC / GW residual / MID+FWD G/A dig — diagnostics only

## Key modules

- `src/markets/cascade_score_v2.py`, `src/markets/short_horizon_generative.py`
- `src/markets/rate_baseline.py`
- `src/models/multistage_mc_milp.py`, `src/models/path_transfer_plan.py`
- `src/models/short_horizon_cascade.py`, `src/models/player_pool.py`
