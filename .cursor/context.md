# Architectural state

Updated: 2026-10-23. **Numagent 2.0** is the product on main (`protocol.numagent_2_0`) — **supersedes Numagent 1.0**. Parked hypotheses: `.cursor/ideas.md`. Stage archaeology: root `CONTEXT.md` and `reports/SUPERSEDED.md`. Canonical index: `reports/LOCKED.md`. Cascade locked index: [`reports/CASCADE_LOCKED.md`](../reports/CASCADE_LOCKED.md).

**Cascade consolidation (2026-10-23):** locked / cascade_default pieces from #101–#106 (rate baseline, joint path, γ=1, FT PE, bench α reject; deadline-causal diagnostic) fast-forwarded onto `cursor/club-expanding-gf-ga-2025-26-cc3d` for landing on `main`.

## Repo freeze sequence (2026-10-20)

| Step | Status | Note |
| --- | --- | --- |
| **1. Freeze product on main** | **done** | #88 (+ #89) merged to `main`. Pre-v2 cascade PR forest #73–#87 **closed** (do not re-open onto main). |
| **2. Archive Numagent 1.0 / `score_xp` in-place** | **done** | Index [`reports/ARCHIVE_NUMAGENT_1_0.md`](../reports/ARCHIVE_NUMAGENT_1_0.md); protocol roles; GW6 entrypoint docstrings; leftover PRs #59/#60/#69/#71/#72 **closed**. Modules stay importable for the gate baseline. |
| **3. Decompose v2 for cascade** | **done** | [`CASCADE_CONTRACT.md`](../reports/CASCADE_CONTRACT.md); `src/strategy/layer4_mu_inject.py` (`pools_from_mu_boards` → `solve_horizon_transfers`). Long-horizon λ→xP stays `cascade_score_v2`. |
| **4. Cascade rebuild plan** | **done** | [`CASCADE_PLAN.md`](../reports/CASCADE_PLAN.md): **cascade_short** / **cascade_long**, shared [`player_pool`](../src/models/player_pool.py). Scorer gate: [`SCORE_XP_V2_CHECKLIST.md`](../reports/SCORE_XP_V2_CHECKLIST.md) — confirm before path-V screens. |
| **5. cascade_short generative** | **done** | `protocol.cascade_short_generative` locked 2026-10-20 (PR #96 / #97): centreline `M`→`k∼M` (**no λ ±15% jitter**)→lineup→attack→CS/GC; soft priors = `score_xp_v2` helpers; firm XI deferred. |
| **6. cascade_long Opta generative** | **PI-approved 2026-10-24** | `protocol.cascade_long_match_outcome`: rankmean±SD latents (`centre_blend=0.5`) → assessed expanding ḡ → λ → `M\|λ` → `k∼M` (hierarchical; no second λ jitter; OLS/`HOME_ADV=1.08` retired); player stack = cascade_short generative **without props**. |
| **7. Cascade viability overnight** | **done** | `cascade_viability_batch` A/D green; C FT+cap smoke OK; B1/B2 gen MC-mean **clearly_loses** vs centreline mean (Δ≈−0.09); B3 inconclusive. Gen-as-mean **parked** — `reports/cascade_short_long_viability_results.md`. |
| **8. Mean t:t+1 vs short gen** | **done** | N_MC=20 product horizon: still **clearly_loses** (Δ≈−0.054). `reports/cascade_short_mean_tt1_vs_gen.md`. |
| **9. Path-V gen vs mean MILP** | **done (diagnostic)** | FT+cap N=20: both solve; gen hits≈7.2 vs mean≈1.6; top-15 Jaccard≈0.36. Not a path-V promote. `reports/cascade_pathv_gen_vs_mean.md`. |
| **10. Minutes on gen paths** | **locked** | `DEFAULT_LINEUP_COV=False` — freeze as-of mean / live xmi. FFScout = pool WHO only. `reports/cascade_live_minutes_layers.md`. |
| **11. Hist MC freq vs mean MILP** | **done (diagnostic)** | 2025-26 N=12: oracle ranks + LOSO GF/GA path vs priced t:t+1 mean. After rate baseline: Haaland top-cap 0.46 (Cash pathology gone). `reports/cascade_mc_vs_mean_hist.md`. |
| **12. Historical rate baseline** | **cascade default** | Start at prior (player else positional); update with each completed GW. `src/markets/rate_baseline.py` · `protocol.score_xp_v2_rate_baseline_prior`. Full mu_post gate deferred. |
| **13. Mean vs path GW residual** | **done (diagnostic)** | Full-side shares; semi-oracle buy-only. 2025-26: mean/path actual ≈647/646; resid bias +12.7 / +1.1. `reports/cascade_mean_path_gw_resid.md`. |
| **14. Top MID+FWD G/A mismatch** | **done (diagnostic)** | Top xp decile: g/EG≈0.85 (pot≈0.82, shares OK); a/EA≈0.61 (pot≈0.72 **and** sA too fat). `reports/top_mid_fwd_ga_mismatch.md`. |
| **15. Path FT potential energy** | **default on (diagnostic)** | `U(b)=ν·b`, ν=`HIT_COST` (+4/FT, +20 at cap). N=12: schedule Δ small (terminal ft 1.00→1.08). `reports/cascade_ft_pe_path.md`. |
| **16. Path every-GW transfers** | **locked** | Joint path plan schedules every horizon GW (`cascade_joint_path_plan`). Hard must-spend: `e=max(0,k−f)`. |
| **17. Cascade γ** | **locked γ=1** | Removed climb `γ=0.9` contamination from cascade/multistage defaults (`cascade_gamma`). |
| **18. Deadline-causal minutes** | **done (diagnostic)** | Oracle m only at deadline GW; as-of for later; rolling replan. N=12: +hits, actual +12, terminal FT flat. `reports/cascade_deadline_causal_minutes.md`. |
| **19. Path bench weight α** | **rejected** | α·squad form screened then removed from path objective. `reports/cascade_path_bench_weight.md`. |

## Recent diagnostic (2026-10-25)

**CS thread closed.** XI CS underprint = early-season selection luck; league
DC calibrated. Do not retune CS.
`reports/xi_cs_selection_hypothesis.md` · windows · `dixon_coles_team_cs_calib.md`.

**G overprint.** Full-share OL gen ~+11. Pot cleared; not dice/k∼M.
**π∝r locked 2026-10-27** — goal shares drop minutes from weights
(m=gate only); assists still ∝rA·m/90. Screen: XI resid **+10.2 → +0.2**
(N=40). Shrink parked as unwanted.
`reports/cascade_share_no_renorm_minutes.md` · `cascade_g_overprint_dig.md`.

**Strand default:** generative `run_strand` / `_undiscounted_scores_by_gw`
require `share_pool` (full club m>0). Live + hist path/mean callers wired;
selection stays thin. Goal weight: `goal_share_weight` (π∝r).

Multi-seed XI residuals (boards once): CS stayed negative across opens;
G stayed positive — `cascade_vs_l3_component_xp_multiseed.md`.

## Plan

Pre-deadline squad advisor under 2026/27 rules. Cursor implements and screens; Gemini proposes formulae; the PI reads gated batches. A result needs the as-of audit and paired intervals on 2022-23…2025-26 (`write_gated_report`). **Numagent 2.0 pots** use The Odds API (AH+OU+BTTS; quota-aware — `.cursor/rules/odds-api-quota.mdc`). Classic climb `gw_lines` may use Betfair. The live 2026/27 holdout is frozen — do not fit on it.

## Locked stack — Numagent 2.0

| Layer | What | Status | Entry |
| --- | --- | --- | --- |
| **Scorer** | `score_xp_v2` (**primary**; `score_xp` superseded) | locked 2026-10-13 · primary 2026-10-19 | `src/markets/numagent_xp.py` · `reports/score_xp_v2.md` · method lock `reports/numagent_2_0_method.md` |
| **Rates** | \(M_0=270\), \(c=0.25\) | promoted 2026-10-16 | `reports/score_xp_v2_rate_m0.md` |
| **Shares** | share-SE → position×minutes prior; **goal π∝r** (m=gate; assists ∝r·m/90) | prior 2026-10-17 · π∝r 2026-10-27 | `reports/score_xp_v2_share_se_pos_prior.md` · `cascade_share_no_renorm_minutes.md` |
| **μ_post** | ≡ `score_xp_v2` (shrink `mode=none`) | locked 2026-10-16 | `protocol.strategy_layer2_shrinkage` |
| **Transfers** | L3 single-deadline | locked | `src/strategy/layer3_week.py` |
| **Horizon** | L4 μ_post priced, `BOOKIE_DEPTH=2`, γ=1 | **strategiser default** | `src/strategy/layer4_week.py` · `reports/layer4_mupost_gamma1.md` |
| **Live pick** | `numagent_live_pick` (props **off**) | GW7 PI lock | `src/live/numagent_live_pick.py` |
| **cascade_short** | priced t:t+1 generative: centreline `M`→`k∼M` (no λ jitter) | locked 2026-10-20 | `cascade_short` / `short_horizon_generative` · `protocol.cascade_short_generative` |
| **cascade_long** | rankmean±SD latents; assessed ḡ; hierarchical λ→`M\|λ`→`k∼M`; generative xP, props off | **PI-approved 2026-10-24** | `cascade_long` / `multistage_mc_milp` · `protocol.cascade_long_match_outcome` |
| **cascade rates** | historical player/positional rate baseline | cascade_default | `src/markets/rate_baseline.py` · `protocol.score_xp_v2_rate_baseline_prior` |
| **cascade path** | joint every-GW FT plan; γ=1; FT **bank pot** (not in \(V\)); minutes frozen | locked / default-on | `path_transfer_plan` · `cascade_joint_path_plan` · `cascade_gamma` · `DEFAULT_LINEUP_COV=False` |
| **transfer-rule screens** | `xi_ft` — joint FT, no captain double; bank pot separate | active 2026-10-24 | `protocol.cascade_path_transfer_only` |
| **Saka→Groß organic probe** | mean boards xi_ft; 7 FT / 0 hits across N | diagnostic 2026-10-24 | `scripts/saka_gross_swap_probe.py --n-strands` |
| · N=12 | open has Groß; Saka out GW10 | done | `saka_gross_swap_probe.json` |
| · N=100 | open has Groß; Saka out GW9 | done | `saka_gross_swap_probe_n100.json` |
| · N=500 | open omits Groß; **buy Groß GW10**, Saka out GW11 | done | `saka_gross_swap_probe_n500.json` |
| **Mean xi_ft hist 2025-26** | oracle ranks + LOSO GF/GA + prior rates; GW8–19 N=500 → **12 FT / 0 hits** | diagnostic 2026-10-24 | `src/eval/cascade_mean_xi_ft_hist.py` · `reports/cascade_mean_xi_ft_hist.md` |
| **Gen-mean convergence** | large N → generative E[xP], **not** centreline (MAE plateaus ~0.36) | diagnostic 2026-10-24 | `src/eval/cascade_mean_strand_convergence.py` |
| **Mean xi_ft rolling stick** | GW8–10 receding replan vs OL; J̄=0.88; soft FT spend; 0 hits | diagnostic 2026-10-24 | `src/eval/cascade_mean_xi_ft_rolling.py` |
| **Mean xi_ft G/CS/GC resid** | OL XI: G +230 / CS +22 / GC −5 (Σ xP−P); roll GW8–10 G over, CS under | diagnostic 2026-10-24 | `src/eval/cascade_mean_xi_ft_component_resid.py` |

**Numagent 1.0 and `score_xp` are superseded** (2026-10-19). Keep 1.0/`score_xp` only for historical climb replay and as the named gate baseline. New screens and the live path are Numagent 2.0 (`score_xp_v2` + L4).

### Cascade / short-horizon alignment (2026-10-19)

Today’s pre-merge cloud cascade work initially scored with **`xp_on_pot` /
`compute_xp`**. Audit: `reports/cascade_pre_numagent2_audit.md`.

**Aligned + re-run on `score_xp_v2`:**

- Short-horizon advisor — `src/models/short_horizon_cascade.py` (L4 corr 1.0;
  `keep_scenario_advisor`). Priced scenario strands: centreline `M` →
  categorical `k∼M` (**λ Uniform±15% retired**) → lineup cov → attack dice →
  CS/GC from rolled GA; soft priors = locked `score_xp_v2` helpers
  (`short_horizon_generative`; firm XI later; bonus-from-roll parked). Mean
  layer stays `score_team_side`.
- Long-horizon Opta multistage (`squad_sum` / `xi_plus_captain` /
  `xi_plus_captain_ft` ± minutes overlay) via
  `src/markets/cascade_score_v2.py` — Opta place empirical GF/GA mean±SD →
  `M` → `k∼M` (`DEFAULT_RANK_METHOD=rankmean`; OLS only via `method="ols"`).
  **`xi_plus_captain_ft`** = joint multi-GW transfer plan
  (`src/models/path_transfer_plan.py`): roll horizon → per-GW scores → one
  MILP schedules opening 15 + holds/transfers/hits (not myopic snapshot FT).
- Oracle-rank cascade 2025/26 — **viable**; oracle − market log **+0.0187**
- Match-band / player G/A finish / rank tracker — rewired + rejudged 2026-10-19
  (all three **park** under `score_xp_v2`; keep baseline z=1.0 / blend=0.5,
  share×λ rates; live place→GF/GA is rankmean, not OLS)

Suite write-up: `reports/long_horizon_numagent2_rerun.md`.
Screens: `reports/match_band_calibration.md`, `reports/player_ga_finish.md`,
`reports/rank_tracker.md`.

**Missed-screen follow-up (2026-10-20)** — all park / plumbing-only:

- Share seeding — `keep_frozen_shares` (Δlog **−0.0131** [−0.0158, −0.0103])
- Assist α_A = team xA/xG — `keep_pooled_alpha_a` (Δlog **−0.00175** [−0.00255, −0.00097]); legacy `score_xp` promote does not transfer
- Priced→Opta strand hand-off — `plumbing_only_no_likelihood_arm` (`reports/strand_handoff_smoke.md`)

### Scorer defaults (live + historical assemble)

- **Decomposed match layer:** `match_outcome_probs` (bookie AH/OU/BTTS → \(M\)) and `match_outcome_roll` (draw inside \(M\)) — `src/markets/match_outcome.py` · `reports/MATCH_OUTCOME.md`. Call-site rewire pending.
- **Decomposed player attack:** `player_attack_from_m` — \(G=\alpha_G xG_M\), \(A=\alpha_A xG_M\), \(E[G_i]=sG_i G\), \(E[A_i]=sA_i A\) — `src/markets/player_attack.py` · `reports/PLAYER_ATTACK.md` (Saka GW20 walkthrough). Shares stay in `numagent_xp`.
- AH+OU (+ BTTS when priced live) → Dixon–Coles \(M\) → pots \(G=\alpha_G xG_M\), \(A=\alpha_A xG_M\)
- α from `fpl_score_rates` (2025/26 prior + live counts; refresh after each completed GW)
- Same-season-only \(rG/rA\) (no prior-season share fill); RATE_M0=270
- Minutes: packet/tags → `xmi`; predicted-XI roles (`lineup:xi` / `lineup:rotated`)
- Share-SE shrink on; prior `position_minutes`; **anytime props off** by default; formula = **props shape π, M levels** (assists no overlay; lift-only parked)
- xCS / xGC additive from the same matrix; saves / DefCon / bonus / cards as in `reports/score_xp_v2.md`
- Gate vs `score_xp` (oracle minutes, GW5–38, m>60): Δ **+0.0531** [+0.0453, +0.0609] — `reports/score_xp_v2_gate.md`

### Strategiser defaults

- L1 joint MILP on absolute μ; L2 identity attach; L3 transfers+hits; **L4 priced path** week-t = L3 board, t+1 = projected `score_xp_v2` on book pots only (never an unpriced week)
- L5 beam screened, not promoted (K=8 locked as beam width only) — L4 stays default
- Captain = max μ; stack-vs-diversify diagnostic only (`reports/fixture_planning_glue.md`)

### Live path

```
python -m src.live.numagent_live_pick --gw N
python -m src.live.numagent_live_pick --gw N --fetch-odds   # Odds API, region=us
```

Prefer `--from-snapshot` / `data/scratch/gwNN_odds_*.json` when the book is already pulled. `--fetch-odds` refreshes AH+OU+BTTS for Dixon–Coles pots (props off unless `--with-props`). Classic climb `gw_lines` readers may still use Betfair folders; frozen `data/live/gw_lines.csv` is not a live book.

### Latest PI lock — GW7 (2026-10-17)

[`data/predictions/2026-27/gw07/numagent_lock.json`](../data/predictions/2026-27/gw07/numagent_lock.json)

- 1 FT Davis → Affengruber · C Haaland · VC Mbeumo (PI)
- Bench: Verbruggen → Murillo → Khalaili → Mainoo
- Model: `score_xp_v2` + `share_se_pos_prior` + L4 depth-2 γ=1, props off · H-obj 123.05
- Report: `reports/numagent_live_pick_gw07.md`

## Climb constants (still in force)

Hold 1.25, switch penalty 1.0, buy gate three prior appearances and xmi≥45. Empty chip map for legacy climb replay. Constants: `src/rules/fpl_2026.py`. Climb modules that still score on `score_xp` are historical only — do not extend them; new work uses `score_xp_v2` / L4.

## Parked (do not revive without a new protocol arm)

Full list: `.cursor/ideas.md` and `reports/LOCKED.md` §Parked.

Highlights: anytime props as live default; share-SE all-minutes / scorers-only; minutes decay ρ=0.7; L4 risk/chase/window/beyond-t+2 form fills; L4 FT-DP / chip half-stop; L5 as strategiser; constant-K μ_post shrink; Betfair pot source on US cloud (geo-blocked — Mac worker only).

## Rules still in force

- Gates: as-of audit + four-season paired intervals before any claim
- Holdout freeze hashes untouched
- Odds API: estimate cost first; region=`us`; prefer snapshots (quota rule)
- Fewer comparisons; objective-function search parked; captain/chip sims deferred until a pre-registered likelihood arm asks
- After a discrete sub-task: write `REPORT_STATE.json`
