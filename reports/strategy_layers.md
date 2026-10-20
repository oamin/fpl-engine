# Strategiser — Numagent 2.0 layered build plan

Part of **Numagent 2.0** (`protocol.numagent_2_0`; supersedes Numagent 1.0).  
`score_xp_v2` is the engine primary scorer and is **not** modified here. (`score_xp` superseded 2026-10-19.)  
**Governing rule:** maximising expected points overrides everything on the default (neutral) posture. Path variation GW1→38 is enough for percentile uniqueness; do not cut μ for ownership optics on the default objective.

**Architecture class (locked intent):** model-based planning — world model → rolling-horizon MILP → chip calendar → small parameter search. Not RL policy-from-scratch. Learning = &lt;10 named knobs tuned under a pre-registered eval, not unconstrained transfer policies.

## Layer map

| Layer | Name | Status | In | Out until later |
| ---: | --- | --- | --- | --- |
| **1** | Single-week joint MILP | ✅ | 15 + XI + captain on absolute μ | — |
| **2** | μ_post | ✅ locked `mode=none` | μ_post ≡ `score_xp_v2` (identity) | η·SE / ownership / constant-K |
| **3** | Transfers + hits | ✅ | Held 15 → this-deadline transfers, FT bank, −4 hits | chips |
| **4** | Rolling horizon | ✅ **default** | μ_post priced, BOOKIE_DEPTH=2, γ=1, hard stop | chips; beyond-t+2 form fills parked |
| **5** | Chip + FT foresight | screened | Beam **K=8** locked; vs L4 covers 0 — **not promoted** | play thresholds |
| **6** | Scenarios / risk | pending | Correlated draws; CVaR/upside; 3 postures | RL policy |
| **7** | Param search + sandbox | parked | Knob search stays deferred per protocol | — |

**Scrapped / parked:** ownership tilt; η·SE; constant-K shrink; L4 risk/chase/window/beyond-form; L5 as default. Index: `reports/LOCKED.md`.

---

## Done

### Layer 1 — single-week MILP
`src/strategy/layer1_week.py` · audit `reports/strategy_layer1_audit.md`

### Layer 2 — μ_post
Protocol `strategy_layer2_shrinkage` (revised 2026-10-16): **`mode=none`** — μ_post ≡ `score_xp_v2`.  
Constant-K historical screen frozen in `reports/se_credibility.md` / `reports/shrink_climb.md` (superseded for decisions).

---

## Proposed build order (work through in sequence)

### Step A — Layer 3: single-deadline transfers ✅

**Code:** `src/strategy/layer3_week.py` · `solve_week_transfers`  
**Tests:** `tests/test_strategy_layer3.py` (5)  
**Audit:** `reports/strategy_layer3_audit.md` (6/6 PASS)  
**Protocol:** `strategy_layer3_transfers` (2026-10-14)

Objective: `Σ μ_post·start + Σ μ_post·captain − 4·hits`.  
Affordability via bank + `sell_price`; max transfers = FT + `max_hits` (default 2).

**Climb sanity:** `python -m src.strategy.layer3_climb` → `reports/layer3_climb.md`  
Pooled shrink−raw **−2.40/GW** [−4.11, −0.76] (below 0). Both arms over-hit under a
one-week objective; shrink hits less but still trails. Does **not** overturn L2 forecast
lock — motivates Layer 4 horizon / hold discipline.

---

### Step B — Layer 4: rolling horizon + terminal value  ✅

**Locked 2026-10-15** — `protocol.strategy_layer4_horizon`

**Construct**
- Transfers + hits only at **t=0** (L3 block); XI+captain each week in H on that fixed 15.
- Commit this week only; re-solve next deadline.
- γ=0.9; terminal μ weight 0.25 on squad at H−1; value credit 0.
- H default 4 (candidates 4/5/6).

**Deliverables**
- `src/strategy/layer4_week.py` · `solve_horizon_transfers`
- Audit **6/6 PASS** (`reports/strategy_layer4_audit.md`) — H=1 ≡ L3
- Climb path=`repeat`: L4−L3 +0.59/GW covers 0 (`reports/layer4_climb.md`)
- Cov-as-μ bookie trial: **parked** (lost to L3) — `reports/layer4_bookie_trial.md`
- **μ_post priced path** (`reports/layer4_mupost_gamma1.md`): week-t = L3 board; t+1/t+2 projected+shrink on FD; hard stop; **γ=1** (undiscounted); unconstrained hits — L4−L3 **+3.56/GW** [+1.65, +5.64]. **Default L4 path.** Prior γ=0.9 under same hits: +2.55/GW (`reports/layer4_mupost_priced.md`, frozen).
- **Fresh template open** (`reports/layer4_mupost_fresh_open.md`): same L3/L4 priced policies, open from crowd GW1 **template** (shared hold) — L4−L3 **+1.87/GW** [−0.35, +4.16] covers 0 → **inconclusive**; flips 2024-25 to +133 (μ-WC was −40). Default stays μ-WC open; gap is path-seed sensitive.
- **Risk-seeking mu_dec** (`reports/layer4_mupost_risk.md`): priced μ_post + λ·(z se_post + z cov SD); +1.48/GW covers 0 → **parked** (not default).
- **Windowed μ_post** (`reports/layer4_mupost_window.md`): W=3/5 rates+moments lose hard to expanding (−10.8 / −7.1 vs L3) → **parked**.
- **Chase upside semi-SD** (`reports/layer4_mupost_chase.md`): always-on clears L3 (+2.34) but −0.56 vs expanding (covers 0) → keep as **optional chase posture**, not default.
- **Adaptive λ vs L3 gap** (`reports/layer4_mupost_posture.md`): chase+cons loses to expanding (−1.93) → **parked**.
- **Chase-only when behind** (`reports/layer4_mupost_chase_behind.md`): clears L3 (+2.85); −0.04 vs expanding (covers 0) → **optional**, not default.
- **Beyond-t+2 A/B/C** (`reports/layer4_mupost_beyond.md`): A hard-stop stays default; B FDR / C decay do not beat A.
- **Beyond-t+2 arm D team-form** (`reports/layer4_mupost_beyond_d.md`): mean(priced)×raw expanding club FDR; D−A **−2.27/GW** clear lose → **park fill; keep A**.
- **Beyond-t+2 D adapt** (`reports/layer4_mupost_beyond_d_adapt.md`): as-of amp / amp·ρ^k; both lose to A (−2.58 / −1.86) → **park**.
- **Team-form terminal** (`reports/layer4_mupost_terminal.md`): same H=3 + hold credit; T−A **−3.27/GW** → **park**. Beyond form stays out of the MILP.

**Locked decision board:** A hard-stop on priced **t:t+1** (`BOOKIE_DEPTH=2`). Beyond form = diagnostic only (`src/strategy/team_form_diagnostic.py`).

**Next:** live shadow of L4; L5 chips only under a new pre-registered arm. Beyond-t+2 boards/terminals parked.

---

### Step C — Layer 5: chip calendar

**Revised architecture (2026-10-15):** strategy must **emerge** from joint chip timing + FT/transfer path on multi-week priced boards. No early-WC / BB-on-DGW hand rules; no play thresholds.

- **Step 1 ✅** — `ChipWallet` legality.
- **Step 2 ✅** — thin TC/BB scoring identity (`score_commit_week`).
- **Step 3 ✅** — WC/FH state (`advance_planner_state`: WC keeps rebuild, FH reverts, FT preserved).
- **Step 4 ✅ (K screen)** — joint beam; prior default K=2 (`reports/layer5_beam_k.md`).
- **Step 5 ✅** — L5 (K=2, myopic) vs L4: **−2.82/GW** covers 0 (`reports/layer5_vs_l4.md`). **Not promoted.**
- **Horizon foundation ✅** — option 3 boards; L5_cal not promoted. Diagnose: myopic beam transfers (−441) (`reports/layer5_nochips_diagnose.md`).
- **Horizon MILP in beam ✅ (2026-10-16)** — depth-0 none/TC/BB = L4 `solve_horizon_transfers`; later depths XI-only.
- **Re-screen ✅ (2026-10-16)** — K=2 **+0.11/GW** covers 0 (`reports/layer5_vs_l4_rescreen.md`). **K grid** → PI lock **`DEFAULT_BEAM_K=8`** (≡16, +91 pooled; still covers 0 vs L4) (`reports/layer5_vs_l4_k_rescreen.md`). **L4 stays strategiser default.**

Audit: `reports/strategy_layer5_audit.md`.

---

### Step D — Layer 6: scenarios + risk posture

**Planning-glue screen (2026-10-15):** `reports/fixture_planning_glue.md`  
Captain haul / CVaR vs μ **inconclusive** (intervals cover 0) → do **not** override captain with draw tails yet.  
Same-fixture stack vs diversify **+0.55** [+0.26, +0.86] — keep as a stack diagnostic; not wired into MILP.

**Construct (remaining)**
- World-model draws: team/match factors, compositional shares, minutes — correlated, not delta-method SE alone.
- Captain / path under scenarios; optional CVaR or upside term (revisit only with a new dated protocol arm).
- Risk dial λ from rank percentile × weeks left → three postures: **consolidate / neutral / chase**.
- Default = neutral = E[μ_post] − hits. Ownership caps only under chase (optional), never as a permanent μ tax.

**Deliverables**
- Draw store (reuse for SE propagation).
- Posture rules (deterministic first).
- Screen: scenarios help captain/path vs mean-only on held-out GWs.

---

### Step E — Layer 7: sandbox + parameter search

**Construct**
- Season sandbox: rules-perfect FPL physics + world-model sampled outcomes/prices (history calibrates the generator; sandbox is not history-free).
- Knobs (~hit threshold / hold margin / γ / H / risk schedule / chip margins): &lt;10.
- Tune with **common random numbers** (CMA-ES or Bayesian opt).
- Validate on held-out real seasons, then live shadow. Sim is for search, not proof.

**Deliverables**
- `src/strategy/sandbox/` (or similar) env API: state → action → next state + reward.
- Pre-registered eval: paired-by-GW vs hold / greedy-1FT / template + per-decision regret.
- Protocol: which knobs are free, search budget, validation seasons.

---

### Parallel

| Item | Note |
| --- | --- |
| Multi-week σ_model for SE_total | Improves L6 later; L2 stays `mode=none` until a dated screen beats identity |
| **Correlated draws (started)** | `src/markets/correlated_draws.py` — match λ shocks + Dirichlet shares; tests green. Feeds L6 / horizon risk. Report `reports/correlated_draws.md` |
| LLM news → minutes schema | Live path only; never chooses transfers |

---

## Evaluation rules (all layers)

- Primary: paired-by-gameweek (or player-GW where relevant), not season totals alone.
- Baselines: hold, one-week greedy / L3-only, template where applicable.
- Every new knob or objective term → dated `experiments/protocol.json` entry **before** reading closed totals.
- 2026/27 holdout stays frozen; do not fit on it.

## Soft vs hard constraints

- **Hard:** FPL rules only (`fpl_2026`).
- **Soft / named knobs:** candidate top-N, H, γ, hold margin, hit willingness, terminal weights, risk dial, chip margins. Tunable under Layer 7 — not silent magic numbers.
- **Not first:** plan-switching penalty, ownership in default objective, RL transfer policy.
