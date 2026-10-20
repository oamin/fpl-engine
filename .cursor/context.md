# Architectural state

Updated: 2026-10-20. **Numagent 2.0** is the product on main (`protocol.numagent_2_0`) — **supersedes Numagent 1.0**. Parked hypotheses: `.cursor/ideas.md`. Stage archaeology: root `CONTEXT.md` and `reports/SUPERSEDED.md`. Canonical index: `reports/LOCKED.md`.

## Repo freeze sequence (2026-10-20)

| Step | Status | Note |
| --- | --- | --- |
| **1. Freeze product on main** | **done** | #88 (+ #89) merged to `main`. Pre-v2 cascade PR forest #73–#87 **closed** (do not re-open onto main). |
| **2. Archive Numagent 1.0 / `score_xp` in-place** | **done** | Index [`reports/ARCHIVE_NUMAGENT_1_0.md`](../reports/ARCHIVE_NUMAGENT_1_0.md); protocol roles; GW6 entrypoint docstrings; leftover PRs #59/#60/#69/#71/#72 **closed**. Modules stay importable for the gate baseline. |
| **3. Decompose v2 for cascade** | **done** | [`CASCADE_CONTRACT.md`](../reports/CASCADE_CONTRACT.md); `src/strategy/layer4_mu_inject.py` (`pools_from_mu_boards` → `solve_horizon_transfers`). Long-horizon λ→xP stays `cascade_score_v2`. |
| **4. Cascade rebuild plan** | **done** | [`CASCADE_PLAN.md`](../reports/CASCADE_PLAN.md): **cascade_short** / **cascade_long**, shared [`player_pool`](../src/models/player_pool.py). Scorer gate: [`SCORE_XP_V2_CHECKLIST.md`](../reports/SCORE_XP_V2_CHECKLIST.md) — confirm before path-V screens. |
| **5. cascade_short generative** | **done** | `protocol.cascade_short_generative` locked 2026-10-20 (PR #96): scenario `k∼M`→lineup→attack→CS/GC from roll; soft priors = `score_xp_v2` helpers; firm XI deferred. |
| **6. cascade_long Opta generative** | **in progress** | `protocol.cascade_long_match_outcome`: Opta rank→GF/GA→λ→`M`→`k∼M`; player stack = cascade_short generative **without props**. |

## Plan

Pre-deadline squad advisor under 2026/27 rules. Cursor implements and screens; Gemini proposes formulae; the PI reads gated batches. A result needs the as-of audit and paired intervals on 2022-23…2025-26 (`write_gated_report`). **Numagent 2.0 pots** use The Odds API (AH+OU+BTTS; quota-aware — `.cursor/rules/odds-api-quota.mdc`). Classic climb `gw_lines` may use Betfair. The live 2026/27 holdout is frozen — do not fit on it.

## Locked stack — Numagent 2.0

| Layer | What | Status | Entry |
| --- | --- | --- | --- |
| **Scorer** | `score_xp_v2` (**primary**; `score_xp` superseded) | locked 2026-10-13 · primary 2026-10-19 | `src/markets/numagent_xp.py` · `reports/score_xp_v2.md` · method lock `reports/numagent_2_0_method.md` |
| **Rates** | \(M_0=270\), \(c=0.25\) | promoted 2026-10-16 | `reports/score_xp_v2_rate_m0.md` |
| **Shares** | share-SE → position×minutes prior | promoted 2026-10-17 | `reports/score_xp_v2_share_se_pos_prior.md` |
| **μ_post** | ≡ `score_xp_v2` (shrink `mode=none`) | locked 2026-10-16 | `protocol.strategy_layer2_shrinkage` |
| **Transfers** | L3 single-deadline | locked | `src/strategy/layer3_week.py` |
| **Horizon** | L4 μ_post priced, `BOOKIE_DEPTH=2`, γ=1 | **strategiser default** | `src/strategy/layer4_week.py` · `reports/layer4_mupost_gamma1.md` |
| **Live pick** | `numagent_live_pick` (props **off**) | GW7 PI lock | `src/live/numagent_live_pick.py` |
| **cascade_short** | priced t:t+1 generative scenarios | locked 2026-10-20 | `cascade_short` / `short_horizon_generative` · `protocol.cascade_short_generative` |

**Numagent 1.0 and `score_xp` are superseded** (2026-10-19). Keep 1.0/`score_xp` only for historical climb replay and as the named gate baseline. New screens and the live path are Numagent 2.0 (`score_xp_v2` + L4).

### Cascade / short-horizon alignment (2026-10-19)

Today’s pre-merge cloud cascade work initially scored with **`xp_on_pot` /
`compute_xp`**. Audit: `reports/cascade_pre_numagent2_audit.md`.

**Aligned + re-run on `score_xp_v2`:**

- Short-horizon advisor — `src/models/short_horizon_cascade.py` (L4 corr 1.0;
  `keep_scenario_advisor`). Priced scenario strands: generative
  `k∼M` → lineup cov → attack dice → CS/GC from rolled GA; soft priors =
  locked `score_xp_v2` DefCon/bonus/cards/saves helpers (`short_horizon_generative`;
  firm XI later; bonus-from-roll parked). Mean layer stays `score_team_side`.
- Long-horizon Opta multistage (`squad_sum` / `xi_plus_captain` /
  `xi_plus_captain_ft` ± minutes overlay) via
  `src/markets/cascade_score_v2.py`
- Oracle-rank cascade 2025/26 — **viable**; oracle − market log **+0.0187**
- Match-band / player G/A finish / rank tracker — rewired + rejudged 2026-10-19
  (all three **park** under `score_xp_v2`; keep baseline z=1.0 / blend=0.5,
  share×λ rates, OLS rank→GF/GA map)

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
