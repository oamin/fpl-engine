# Lock — L3 centreline forecasting model

**Locked:** 2026-10-30  
**Protocol:** `l3_centreline_forecast`  
**Status:** locked (product forecast μ)

Front door for **how we forecast player-GW expected points**. Discrete CS/GC
likelihood is a separate eval lock (`l3_discrete_cs_gc_likelihood`).

## Product forecast (μ)

| Horizon | λ / match layer | Player board |
| --- | --- | --- |
| **Near / priced** (`t`, `t+1`) | Live **Odds API** AH+OU+BTTS (hist: Football-Data AH+OU) → Dixon–Coles \(M\) → moments | `score_team_side` → `score_xp_v2` |
| **Far / unpriced** (`t+2…`) | **Opta** place rankmean, **`venue_mode=blind`**, identity shrinks → λ → \(M\) moments | **same** `score_team_side` |

Freeze at decision GW `t`: rates, shares, alphas as-of `t`. Drive every target
`g ∈ [t, end]` with the λ source above. Product strategiser default horizon is
**H=12** (`t…t+11`): priced near + Opta far. Full-season dig board:
**L3@t + Opta λ** (`protocol.l3_opta_lambda_horizon` ·
`reports/l3_opta_lambda_horizon.md`).

### Locked pieces (already in force — consolidated here)

| Piece | Protocol | Note |
| --- | --- | --- |
| Scorer / mean assembler | `numagent_2_0` / `score_xp_v2` | `src/markets/numagent_xp.py` |
| Attack rates | `score_xp_v2_rate_baseline_prior` = cascade_default | prior + same-season update |
| Share pool | full club `m>0` | no thin-board renorm |
| Goal share weight | \(w = r_G \cdot m/90\) | π∝r parked (log-score clear lose) |
| α_G / α_A | as-of pooled counts | Opta own-λ ×α≠1 parked — keep α=1 |
| Opta club λ | `cascade_long_match_outcome` + `cascade_long_blind_venue_lock` | blind; `AWAY_SHRINK=LAM_SHRINK=HOME_ADV=1` |
| Strategy horizon board | `l3_opta_lambda_horizon` | L3 mechanics + Opta λ for unpriced `g` |
| CS/GC **eval** likelihood | `l3_discrete_cs_gc_likelihood` | Bern/cat from \(M\); does **not** change μ |

## Evidence for this lock

1. **Gen MC-mean as μ clearly loses** vs centreline on the same \(M\)
   (viability B1/B2; t:t+1 N_MC=20). Mean board stays closed-form
   `score_team_side`.
2. **Horizon snapshots:** generative cascade_long @t G MAE ≈0.846 vs
   contemporaneous L3@g ≈0.820 (`cascade_eligible_horizon_snapshot`).
3. **L3@t + Opta λ:** G MAE 0.818 vs L3@g+FD 0.805 — most of the gen miss
   was dice/MC-mean, not Opta λ (`l3_opta_lambda_horizon`, 2025-26).
4. **Opta λ defaults:** blind + identity shrinks PI-locked 2026-10-29 after
   venue/scale/lag/balance screens failed OOS or hurt MAE.
5. **Discrete CS/GC** likelihood promotes under same-skeleton GKP/DEF
   (separate lock 2026-10-30).

## Explicitly not the forecast μ

| Arm | Status |
| --- | --- |
| Generative MC-mean xP as board | parked — clearly_loses |
| Mode / median scoreline as μ | parked — freezes favourites |
| Factored-vs-Gaussian-total eval | parked — incommensurate |
| Opta own-λ temper α≠1 | parked — PI keep α=1 |
| place×venue / expanding H/A / FD venue-scale / margin-shrink / lag-1 / remaining-balance | parked under blind lock |
| Player G/A count likelihood | not yet screened |

## Generative path (still available — not μ)

`cascade_short_generative` / long `k∼M` strands remain for **scenario / path**
work. They do **not** replace centreline μ for scoring or strategy boards.
Any revive of gen-as-mean needs a new pre-registered likelihood arm.

## Modules

| Role | Path |
| --- | --- |
| Mean board | `src/markets/numagent_xp.py` (`score_team_side`) |
| **Strategiser horizon factory** | `src/strategy/l3_centreline_horizon.py` (`build_l3_centreline_horizon`) |
| Live / receding pick | `src/live/numagent_live_pick.py` (`--centreline-horizon`, default 5) |
| FD short fits | `src/models/short_horizon_cascade.py` |
| Opta λ / place | `src/models/opta_gf_ga_horizon.py` |
| Horizon dig / board recipe | `src/eval/l3_opta_lambda_horizon.py` |
| Rate baseline | `src/markets/rate_baseline.py` |
| Live near book | Odds API via `numagent_live_pick --fetch-odds` / snapshots (quota rule) |

## Re-open rule

Dated `protocol.l3_centreline_forecast` update required before changing the
μ recipe (λ source by horizon, assembler, share/α defaults). PI approval if
the change would break a published comparison.
