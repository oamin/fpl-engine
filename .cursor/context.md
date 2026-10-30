# Architectural state

Updated: 2026-10-30. **Numagent 2.0** is the product on main (`protocol.numagent_2_0`) — **supersedes Numagent 1.0**. Parked hypotheses: `.cursor/ideas.md`. Stage archaeology: root `CONTEXT.md` and `reports/SUPERSEDED.md`. Canonical index: `reports/LOCKED.md`. Cascade locked index: [`reports/CASCADE_LOCKED.md`](../reports/CASCADE_LOCKED.md).

**Cascade consolidation (2026-10-23):** locked / cascade_default pieces from #101–#106 (rate baseline, joint path, γ=1, FT PE, bench α reject; deadline-causal diagnostic) fast-forwarded onto `cursor/club-expanding-gf-ga-2025-26-cc3d` for landing on `main`.

## Repo freeze sequence (2026-10-20)

| Step | Status | Note |
| --- | --- | --- |
| **1. Freeze product on main** | **done** | #88 (+ #89) merged to `main`. Pre-v2 cascade PR forest #73–#87 **closed** (do not re-open onto main). |
| **2. Archive Numagent 1.0 / `score_xp` in-place** | **done** | Index [`reports/ARCHIVE_NUMAGENT_1_0.md`](../reports/ARCHIVE_NUMAGENT_1_0.md); protocol roles; GW6 entrypoint docstrings; leftover PRs #59/#60/#69/#71/#72 **closed**. Modules stay importable for the gate baseline. |
| **3. Decompose v2 for cascade** | **done** | [`CASCADE_CONTRACT.md`](../reports/CASCADE_CONTRACT.md); `src/strategy/layer4_mu_inject.py` (`pools_from_mu_boards` → `solve_horizon_transfers`). Long-horizon λ→xP stays `cascade_score_v2`. |
| **4. Cascade rebuild plan** | **done** | [`CASCADE_PLAN.md`](../reports/CASCADE_PLAN.md): **cascade_short** / **cascade_long**, shared [`player_pool`](../src/models/player_pool.py). Scorer gate: [`SCORE_XP_V2_CHECKLIST.md`](../reports/SCORE_XP_V2_CHECKLIST.md) — confirm before path-V screens. |
| **5. cascade_short generative** | **done** | `protocol.cascade_short_generative` locked 2026-10-20 (PR #96 / #97): centreline `M`→`k∼M` (**no λ ±15% jitter**)→lineup→attack→CS/GC; soft priors = `score_xp_v2` helpers; firm XI deferred. |
| **6. cascade_long Opta generative** | **PI-approved 2026-10-24**; **venue blind locked 2026-10-29** | `protocol.cascade_long_match_outcome`: rankmean±SD latents (`centre_blend=0.5`) → assessed expanding ḡ → λ → `M\|λ` → `k∼M` (hierarchical; no second λ jitter; OLS/`HOME_ADV=1.08` retired); **`venue_mode=blind`** + identity `AWAY_SHRINK`/`LAM_SHRINK` locked (PI); player stack = cascade_short generative **without props**. `reports/cascade_long_blind_venue_lock.md`. |
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
**π∝r parked 2026-10-27** — gated log-score clear lose; keep `w=r·m/90`.
Component digs (shared L3 XI, no captain): GW8–12 π∝r ΣG 42 vs actual 44
(CS luck 42 vs 68); **GW20–24** CS calibrated (39.7 vs 42) — early-season
sheet luck cleared.
**Season map GW8–38 π∝r** (decision-gw-only, N=40): casc ΣG 273 vs P 259
(+14; l3 +51); CS 253 vs 258 (−5; l3 −14). MAE casc ≤ l3 on G/CS/GA.
Residuals swing with actual variance (GW20–26 hot G underprint −23; GW27–38
cold G overprint +40) while cascade stays ~8–10 G/GW — not a steady G gap.
`reports/cascade_vs_l3_component_xp_pi_r_gw8-38.md`.
**±10 G swing dig:** haul weeks (5, Σ−60) = 2–3 multi-goal scorers
(Thiago/Rice/Haaland/Ekitiké); blank-XI weeks (4, Σ+44) = P_G=0 vs ~10 xP.
Casc≈l3 on those weeks; player season Σresid near 0. Do not retune shares.
`reports/cascade_g_resid_swings_dig.md`.
**cascade_long Opta decision-GW map** (`--match-source opta`, π∝r GW8–38):
opta ΣG +32.5 / CS −29 vs actual; FD short +14 / −5; l3 +51 / −14.
MAE ~tied with FD. Opta overprints G and underprints CS vs short on this
shared-XI screen — not a promote over FD for decision-GW components.
`reports/cascade_vs_l3_component_xp_pi_r_opta_gw8-38.md`.
**cascade_long horizon dig** (`cascade_long_horizon_component_resid`, π∝r,
t+2…38, pools refresh each t, L3 XI, no captain): horizon Σ resid_G shrinks
with n_plan, but **per-GW** bias stays ~+2…+6 (cell mean +3.7 G / +1.3 CS).
Overprint at every horizon offset — not a late-season rescue.
Overlay **short FD t:t+1** on same XI: mean G/n long **+3.5** vs short
**+1.9**; CS/n long **+1.6** vs short **−0.1**. Short swings ±10; long bias
steadier but higher.
`reports/cascade_long_horizon_component_resid.md`.
**cascade_long G resid dig:** Opta club λ − actual **+0.15** (FD ~0), almost
all on **away (+0.30)**; player resid FWD **+30.5** of +32.5; away played
Σresid +25 vs home +1. Opta−FD casc gap +18.5 = λ venue into FWD under π∝r
(`HOME_ADV=1.0`). Do not retune shares; venue fix needs pre-reg λ screen.
`reports/cascade_long_g_resid_dig.md`.
**Venue λ (PI 2026-10-27):** formula is **away_shrink on λ_A only**
(`λ_A ← λ_A·s`; `λ_H` untouched); **no** 0.20–3.50 λ clip. Symmetric
`HOME_ADV` screen superseded (failed promote by moving bias onto home).
**away_shrink screen:** train fit → **0.86**; holdout candidate 0.93 and fit
**both PASS** (home |bias| flat; away +0.256 → +0.150 / +0.043).
**XI counterfactual** (`--away-shrink 0.86`, Opta π∝r GW8–38): resid G
**+32.5 → +12.1** (FD short +14.0); CS **−29.2 → −11.1** (FD −5.4). Default
`AWAY_SHRINK=1.0` unchanged until PI re-approval.
`reports/cascade_long_away_shrink_lam_screen.md` ·
`reports/cascade_vs_l3_component_xp_pi_r_opta_gw8-38_as086.md`.
**resid vs s trend (GW8–38):** G rises ~−39→+32 and CS falls ~+53→−29 as s:0.5→1.0;
both near 0 around **s≈0.75** (G −2.4 / CS +2.4). Fit 0.86 still +12/−11.
`reports/cascade_away_shrink_xi_resid_trend.md`.
**s=0.75 screen arm:** λ promote **PASS** (away +0.256→−0.123, home flat,
MAE ↓). XI resid G **−2.4** / CS **+2.4**.
`reports/cascade_long_away_shrink_lam_screen_s075.md`.
**Venue verdict (PI 2026-10-27):** keep formula (away-only shrink, **no λ
clip**); default **`AWAY_SHRINK=1.0`**. Fixed `s=0.768` / `lam_shrink`
parked (OOS / holdout FAIL).
**G overprint isolate:** Opta XI resid G **+32.5** = **+18.5 Opta−FD λ
(57%)** + **+14 FD−actual generative/π∝r (43%)**. Opta home λ≈actual;
away **+0.26**, worst on **away favourites (+0.43)**. Place GF/GA are
venue-blind season totals. Naive place×venue diagnostic zeros away bias
but breaks home — needs a proper pre-reg OOS screen, not fixed shrink.
`reports/cascade_long_g_overprint_isolate.md`.
**Place×venue λ path (locked blind):** `venue_mode=blind|venue|
venue_away|naive|blend` on `sample_match_from_opta_ranks` /
`sample_opta_scenario`. Primary candidate was **`venue_away`** (λ_H blind;
λ_A = place×venue away attack × blind home defence).
**Place×venue screen:** holdout **venue_away PASS** / blend_fit PASS
(away +0.256→+0.117, home flat); full `venue`/`naive` FAIL (home).
OOS **1/3** — **`DEFAULT_VENUE_MODE=blind` locked 2026-10-29 (PI)**.
`reports/cascade_long_place_venue_lam_screen.md`.
**FD generative dig:** π∝r resid G **+14**; default `rate_minutes`
**+100** on the same FD map (FWD +47). α_G empty ~0.04 irrelevant.
`rate_minutes` concentrates shares (HHI 0.17 vs 0.15). Live generative
problem is the **default minutes weight**, not the π∝r dig arm.
`reports/cascade_fd_generative_g_dig.md`.
**Club Σλ vs place GF (2026-10-28):** pooled **Σλ − place GF ≈ +0.10**
(design converges). Σλ − actual tracks whether the season under/over-shoots
the LOSO prior (e.g. 2025-26 +5.8, 2023-24 −7.4). Venue-blind: cold
seasons show H~0 / A high. XI heat ≠ escaped GF anchors.
`reports/cascade_long_club_lam_vs_place_gf.md`.
**Cascade goal_weight likelihood dig:** component G log π∝r−rm **+0.015**
covers 0 (undetermined) on FD shared XI; gated `score_xp_v2` π∝r remains
**park_clear_lose**. Default `rate_minutes` unchanged.
`reports/cascade_fd_goal_weight_likelihood_dig.md`.
**Cascade_short gen goal_weight screen (2026-10-28):** eligible-pool
likelihood, four seasons. π∝r − rate_minutes **+0.008** [-0.002, +0.017]
→ **inconclusive** (not promote). Both gen arms lose to L3 centreline
(~−0.03…−0.04). Default `rate_minutes` unchanged — Dig B’s +100 XI resid
does not overturn the gate.
`reports/cascade_short_goal_weight_likelihood_screen.md`.
**Gen vs L3 loss anatomy (2026-10-29):** bias nearly identical (~−0.96);
gen higher pred SD (1.81 vs 1.57) and lower corr (0.217 vs 0.237); Δ log
worst in top L3-xp quintile. Confirms viability B1 — variance/ranking, not
a mean-level G fix. Gen-as-mean stays parked.
`reports/cascade_gen_vs_l3_loss_anatomy.md`.
**Expanding H/A λ screen (2026-10-29):** as-of club H/A rates (PRIOR_N=10)
vs Opta blind. Holdout **expanding_away PASS** (A +0.259→+0.031, H flat);
`expanding_ha` / `blend_fit` FAIL home/away thresholds. OOS **2/4**
(expanding_away) — parked under blind lock.
`reports/cascade_long_expanding_ha_lam_screen.md`.
**FD vs Opta ranking λ (2026-10-29):** 2025-26 GW8–38 n=618 paired.
Opta−FD **+0.068** (MAE 0.306, corr 0.706). vs act: Opta **+0.122** /
FD **+0.054**. Venue: Opta A **+0.255** vs FD A **+0.023**; Opta H flat,
FD H **+0.086**. Favourites / top-rank tercile drive Opta−FD gap.
`reports/cascade_long_fd_vs_opta_lam_compare.md`.
**Home/away Opta−FD drivers (2026-10-29):** one mechanism all seasons — Opta H/A flat vs FD HA≈+0.32; resid≈Opta_level−FD_venue. Size varies with Opta season level + FD away softness (2023-24 away≈0 because FD_A≈Opta_A). `reports/cascade_long_fd_opta_venue_resid_drivers.md`.
**Opta−FD gap all seasons (2026-10-29):** pooled **−0.024** (n=2498) masks sign flips — Opta>FD in 22/25, Opta<FD in 23/24. Home: Opta below FD all four. Away: Opta above FD in 3/4 (not 2023-24). `reports/cascade_long_fd_opta_gap_all_seasons.md`.
**Margin-shrink λ trial (2026-10-29):** strength-mismatch lever `λ'=m+α(λ−m)` — **FAIL** (inert: E[λ_H]≈E[λ_A] under blind so venue biases do not move). `reports/cascade_long_margin_shrink_lam_screen.md`.
**FD venue-scale λ trial (2026-10-29):** `λ'_ha=r_ha·λ_ha` toward FD. Train r_H=1.14 / r_A=0.93. Holdout **away_only PASS**; full fit FAIL home; OOS **2/4** — keep identity. Same OOS class as place×venue / expanding_away. `reports/cascade_long_fd_venue_scale_lam_screen.md`.
**Opta−FD residual drivers (2026-10-29):** form not lean-worthy (univ R²≈0; ΔR² **+0.03** after structure). Dominant: **venue + rank_gap / Opta margin** (R²≈**0.40**). `reports/cascade_long_fd_opta_resid_drivers.md`.
**Form vs Opta−FD residual (2026-10-29):** 2025-26 as-of L5 GD. Raw form flat (confounded with place). Conditional on rank+venue: partial r=**−0.23**; FD moves with form (**+0.24**), Opta does not (**+0.03**). Within top/bottom bands hot−cold resid **≈−0.13**.
`reports/cascade_long_fd_opta_form_resid.md`.
**FD↔Opta scale dig (2026-10-29):** not a global mul (Opta−FD flips
sign 25→24). Implementable **venue mean-ratio**
`λ'_H=r_H·λ_H`, `λ'_A=r_A·λ_A` with `r=mean(FD)/mean(Opta)`:
2025-26 **r_H=1.064 / r_A=0.847**; 2024-25 **1.166 / 0.939**; pooled
**≈1.115 / 0.893**. Hook next to `away_shrink` (+ `home_scale`). Affine
`FD≈a·Opta+b` (a<1) needed for tails. **No default flip** (OOS unstable).
`reports/cascade_long_fd_opta_lam_scale_dig.md`.
**Lag-1 / remaining-balance digs (2026-10-29):** lag-1 Opta−FD tracks book
not ball; lag-1 Opta−actual and remaining-season balance both raise MAE vs
blind. **PI lock:** leave blind Opta as default —
`reports/cascade_long_blind_venue_lock.md`.
**Eligible-pool E[G,CS,GC] vs L3 (2026-10-29):** decision-GW-only dig
**superseded** — wrong vs PI intent (not a t:horizon snapshot).
`reports/cascade_eligible_component_resid.md`.
**Eligible horizon snapshots (2026-10-30):** at each t, freeze cascade
short (FD t:t+1) + long (Opta blind t:end); compare each target g to
**L3@g** (FD near g) and actuals (`m>60`). 31 decisions, n=85601 cells.
long−L3 G ΔMAE **near +0.054 / far +0.021**; short t:t+1 G MAE 0.859 vs
L3 0.814. L3 closer near-term; far gap shrinks (some late offsets long≈L3).
Defaults unchanged. `reports/cascade_eligible_horizon_snapshot.md`.
**L3@t + Opta λ horizon (2026-10-30):** same freeze×target design, but
**L3 centreline** (`score_team_side`) driven by Opta blind centreline λ
(no generative dice). n=89444. G MAE **0.818** vs L3@g+FD **0.805**
(near ΔMAE **+0.013**, far **+0.012**) — much closer than generative long
(≈0.846). CS near ΔMAE **−0.006** (slightly beats L3@g). Promising
strategy-score alternative to cascade gen; defaults unchanged.
`reports/l3_opta_lambda_horizon.md`.


**Strand default:** generative `run_strand` / `_undiscounted_scores_by_gw`
require `share_pool` (full club m>0). Live + hist path/mean callers wired;
selection stays thin. Goal weight default: `rate_minutes`.

Multi-seed XI residuals (boards once): CS stayed negative across opens;
G stayed positive — `cascade_vs_l3_component_xp_multiseed.md`.

## Plan

Pre-deadline squad advisor under 2026/27 rules. Cursor implements and screens; Gemini proposes formulae; the PI reads gated batches. A result needs the as-of audit and paired intervals on 2022-23…2025-26 (`write_gated_report`). **Numagent 2.0 pots** use The Odds API (AH+OU+BTTS; quota-aware — `.cursor/rules/odds-api-quota.mdc`). Classic climb `gw_lines` may use Betfair. The live 2026/27 holdout is frozen — do not fit on it.

## Locked stack — Numagent 2.0

| Layer | What | Status | Entry |
| --- | --- | --- | --- |
| **Scorer** | `score_xp_v2` (**primary**; `score_xp` superseded) | locked 2026-10-13 · primary 2026-10-19 | `src/markets/numagent_xp.py` · `reports/score_xp_v2.md` · method lock `reports/numagent_2_0_method.md` |
| **Rates** | \(M_0=270\), \(c=0.25\) | promoted 2026-10-16 | `reports/score_xp_v2_rate_m0.md` |
| **Shares** | share-SE → position×minutes prior; goal **π∝r·m** (π∝r parked) | prior 2026-10-17 · π∝r park 2026-10-27 | `reports/score_xp_v2_share_se_pos_prior.md` · `score_xp_v2_goal_share_rate.md` |
| **μ_post** | ≡ `score_xp_v2` (shrink `mode=none`) | locked 2026-10-16 | `protocol.strategy_layer2_shrinkage` |
| **Transfers** | L3 single-deadline | locked | `src/strategy/layer3_week.py` |
| **Horizon** | L4 μ_post priced, `BOOKIE_DEPTH=2`, γ=1 | **strategiser default** | `src/strategy/layer4_week.py` · `reports/layer4_mupost_gamma1.md` |
| **Live pick** | `numagent_live_pick` (props **off**) | GW7 PI lock | `src/live/numagent_live_pick.py` |
| **cascade_short** | priced t:t+1 generative: centreline `M`→`k∼M` (no λ jitter) | locked 2026-10-20 | `cascade_short` / `short_horizon_generative` · `protocol.cascade_short_generative` |
| **cascade_long** | rankmean±SD latents; assessed ḡ; hierarchical λ→`M\|λ`→`k∼M`; **`venue_mode=blind`** + identity shrinks; generative xP, props off | **PI-approved 2026-10-24**; **blind locked 2026-10-29** | `cascade_long` / `multistage_mc_milp` · `protocol.cascade_long_match_outcome` · `reports/cascade_long_blind_venue_lock.md` |
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

## Kept — same-skeleton discrete CS/GC likelihood (GKP/DEF, 2026-10-30)

Pre-registered pair **promotes** (Δ log +2.828 [+2.798, +2.858]): discrete
CS/GC from M beats Gaussian CS/GC under identical five-term skeleton on
GKP/DEF. Mean board unchanged. `reports/m_skel_cs_gc_likelihood_screen.md` ·
`protocol.m_skel_cs_gc_likelihood_screen`.

## Parked — discrete M-event likelihood vs Gaussian-on-total (2026-10-30)

Pre-registered factored M-event likelihood **clearly loses** vs Gaussian(total|μ,σ=3)
(Δ log −4.617 [−4.632, −4.602]); incommensurate 5-term vs 1-term score.
Same-skeleton diagnostic (discrete CS/GC only) wins Δ +3.231 [+3.212, +3.250].
Mean layer unchanged. `reports/m_discrete_event_likelihood_screen.md` ·
`protocol.m_discrete_event_likelihood_screen`.

## Parked (do not revive without a new protocol arm)

Full list: `.cursor/ideas.md` and `reports/LOCKED.md` §Parked.

Highlights: anytime props as live default; share-SE all-minutes / scorers-only; minutes decay ρ=0.7; L4 risk/chase/window/beyond-t+2 form fills; L4 FT-DP / chip half-stop; L5 as strategiser; constant-K μ_post shrink; Betfair pot source on US cloud (geo-blocked — Mac worker only).

## Rules still in force

- Gates: as-of audit + four-season paired intervals before any claim
- Holdout freeze hashes untouched
- Odds API: estimate cost first; region=`us`; prefer snapshots (quota rule)
- Fewer comparisons; objective-function search parked; captain/chip sims deferred until a pre-registered likelihood arm asks
- After a discrete sub-task: write `REPORT_STATE.json`
