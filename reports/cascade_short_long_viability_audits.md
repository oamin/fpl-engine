# Cascade short → long viability audits (FT + captain)

Diagnostic design only — **not** a gated promote until each arm is
pre-registered in `experiments/protocol.json` and run under
`write_gated_report` where applicable.

**Stack under test (locked):** centreline `M` → categorical `k ∼ M` (no λ
jitter); cascade_short priced t:t+1 (props when available); cascade_long
Opta place→GF/GA±MAE for t+2…; shared `player_pool`; objective
`xi_plus_captain_ft` (chips off). Contaminated ±15% trial is **out of sample**.

**Development metric:** player-GW log score. Free-transfer climb / path-V
is a **sanity check** after likelihood does not clearly lose, or when the
claim is about the transfer/captain rule itself (`PROJECT_OUTLINE.md`).

---

## Ladder (run in order; stop on hard fail)

### A. Integrity (no likelihood claim)

| # | Audit | Pass if | How |
| --- | --- | --- | --- |
| A1 | **As-of freeze** | Pool rates / priors / rolling m use only GW `< t`; scored GW minutes/points/row existence unused | `gates.run_asof_audit` on the board builder used by short+long |
| A2 | **No λ jitter** | `MARKET_BAND == 0`; priced `sample_priced_scenario(band≠0)` raises; short strands pass `moment_overrides=None` | unit tests (already); smoke one strand JSON has `lam_h == fit.lam_h` |
| A3 | **k ∼ M support** | Every rolled `(hg,ag)` has `M[hg,ag] > 0`; empirical mean goals ≈ `M` moments within Monte Carlo noise | extend `test_scoreline_from_m_*`; N=2k City–Fulham / Arsenal–Everton smoke |
| A4 | **Branch routing** | GW with odds → `source=cascade_short_priced`; later GWs → Opta / `cascade_long_opta` | strand scoreline dump on GW8–19 |
| A5 | **Props scope** | Props only on priced short weeks; long weeks `home_props/away_props` empty | assert in `score_pool_generative_on_scenarios` meta / strand records |
| A6 | **FT / captain legality** | Squads satisfy FPL 2026 formation, budget, ≤3/club; captain ∈ XI; hits = max(0, transfers−FT); FT bank advances between snapshots | MILP post-checks + `fpl_2026` rules; N=12 strand dump |

Hard fail on A1–A2 or A6 → do not open B/C.

### B. Likelihood (player-GW log) — primary viability

Register **one comparison at a time**. Closed seasons only for gated claims;
live 2026/27 holdout frozen (do not fit).

| # | Arm | Treatment | Baseline | Metric | Viable if |
| --- | --- | --- | --- | --- | --- |
| B1 | **Short generative vs mean** | priced t:t+1 generative `k∼M`→lineup→dice→CS/GC | mean `score_team_side` on same centreline `M` | paired GW log-score Δ, 4 closed seasons | does **not clearly lose** (interval not wholly below 0) |
| B2 | **Long Opta generative vs mean-λ** | Opta place→GF/GA→`M`→`k∼M` generative xP | `score_team_side` on Opta centreline λ moments (same ranks) | same | does not clearly lose |
| B3 | **Hand-off continuity** | strand: short priced `k∼M` at t (and t+1 if priced) then Opta for t+2… | Opta-only from t (ignore book) | player-GW log on weeks t…t+H (H≤4 first) | hand-off does not clearly lose vs Opta-only |

Notes:

- Pool = shared `player_pool` (or minutes>60 + AH+OU where that is the
  locked screen pool — do not mix mid-screen).
- Minutes: oracle for historical screens; live tags only on live path.
- B3 is the first arm that claims **short→long** as a single forecast path.

### C. Path-V / FT+captain rule (sanity after B)

Only open if B1–B3 do not clearly lose (or if the claim is purely the
transfer objective).

| # | Arm | Treatment | Baseline | Metric | Viable if |
| --- | --- | --- | --- | --- | --- |
| C1 | **Captain objective** | `xi_plus_captain` on short+long scenario μ | `squad_sum` | realised XI+captain points (or path-V) on closed seasons / hist FD | does not clearly lose vs squad_sum |
| C2 | **FT path** | `xi_plus_captain_ft` (free rebuild at first snapshot; FT/hits later) | weekly free rebuild (`xi_plus_captain` each snapshot, no continuity) | same | FT path does not clearly lose; hit rate not absurd |
| C3 | **Full short→long FT+cap** | priced short boards for t:t+1 + Opta long for t+2… under `xi_plus_captain_ft` | L4 priced γ=1 no-chip (`strategy_layer4_mupost_priced`) on overlapping priced weeks; Opta-only FT path on longer horizon | path-V / realised points where defined | does not clearly lose on priced overlap; longer horizon is diagnostic only until B3 clears |

Diagnostics to log (not win criteria): stick rate, mean hits, mean transfers,
captain concentration (top-1 cap share), Haaland/premium freq vs enablers,
cross-strand scoreline fan-out by horizon offset.

### D. Smoke / regression (cheap, every PR)

| # | Test | Intent |
| --- | --- | --- |
| D1 | `MARKET_BAND == 0`; priced band≠0 raises | no jitter regression |
| D2 | One strand GW8–12, N=1, generative on, FT+cap | finishes;  sources short then long |
| D3 | MILP legal squad counters | formation / captain / hits |
| D4 | Contaminated artifact banner present | `multistage_xi_captain_ft_short_long_trial.md` stays marked |

---

## Recommended first batch (pre-register, then run)

1. **A1–A6** integrity pack (unit + one N=4 strand smoke).  
2. **B1** short generative vs mean (hist FD, closed seasons) — likelihood.  
3. **B2** long generative vs mean-λ (Opta/oracle ranks as already used in
   `oracle_rank_cascade`) — likelihood.  
4. Only then **B3** hand-off likelihood.  
5. **C2** then **C3** as FT+captain sanity (do not promote on frequency tables).

Wall-clock ballpark (this VM): A+D minutes; B1/B2 hours if full 4-season
gate; C3 N=12 generative ~6 min diagnostic (not a gate).

---

## Explicit non-goals

- Season-total winner / `pass_margin`.
- Chips.
- Re-opening share seeding, assist α remap, or λ Uniform bands.
- Fitting on live 2026/27 holdout.
- Citing the contaminated ±15% FT trial as evidence.

---

## Decision language

| Verdict | Meaning |
| --- | --- |
| **viable** | Integrity green; registered likelihood arm does not clearly lose; FT+cap sanity not clearly worse than named baseline |
| **park** | Integrity fail, or likelihood clearly loses, or FT path pathological (e.g. hits dominate V) |
| **plumbing_only** | A/D pass but B/C not yet run — current state of strand hand-off |

Reproduce scaffolding (after protocol entries exist):

```bash
# D / A smoke
PYTHONPATH=. python3 -m unittest tests.test_short_horizon_v2 tests.test_multistage_mc_milp
PYTHONPATH=. python3 -m src.models.multistage_mc_milp \
  --n 4 --snapshot-every 2 --objective xi_plus_captain_ft \
  --no-minutes-overlay --priced-gw 8 \
  --out data/predictions/2026-27/cascade_viability_smoke
```
