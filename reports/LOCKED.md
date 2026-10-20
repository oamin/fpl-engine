# Locked & promoted — Numagent 2.0

**Product on main:** Numagent 2.0 (`protocol.numagent_2_0`). **Supersedes Numagent 1.0** (ep_next freeze / `score_xp` live path). Front door for what won. Everything else in `reports/` is evidence, screen history, or parked arms.

| | Numagent 1.0 (superseded) | **Numagent 2.0 (current)** |
| --- | --- | --- |
| Live score | captured `ep_next` | **`score_xp_v2`** |
| Historical / climb score | `score_xp` | `score_xp_v2` (primary); `score_xp` gate baseline only |
| Pots | opening 1X2 / Betfair `gw_lines` | **Odds API AH+OU+BTTS** (snapshots or `--fetch-odds`) |
| Transfers | `wildcard_plan` / climb search | **L3 → L4** μ_post priced |
| Entry | `deadline` / `wildcard_plan` | **`numagent_live_pick`** |
| Protocol | `numagent_1_0` | **`numagent_2_0`** |

Do not treat stage_* / GW1–5 / stragent / GW6 `ep_next` dry runs as the live path.

## Live decision

| Artifact | Path |
| --- | --- |
| **GW7 PI lock** | `data/predictions/2026-27/gw07/numagent_lock.json` |
| Pick report | `reports/numagent_live_pick_gw07.md` |
| Live entrypoint | `python -m src.live.numagent_live_pick --gw N` |
| Odds refresh | `… --fetch-odds` (region=`us`; quota rule) |

## Scorer (`score_xp_v2` — Numagent 2.0 primary)

**`score_xp` superseded 2026-10-19** (`protocol.primary_scorer`). Gate: vs `score_xp` Δ **+0.0531** [+0.0453, +0.0609].

| Lock | Date | Report | Δ / note |
| --- | --- | --- | --- |
| Primary scorer | 2026-10-13 · primary 2026-10-19 | `score_xp_v2.md`, `numagent_2_0_method.md`, `score_xp_v2_gate.md` | replaces `score_xp` |
| RATE_M0=270 | 2026-10-16 | `score_xp_v2_rate_m0.md` | +0.0037 [+0.0006, +0.0071] |
| Share-SE position×minutes | 2026-10-17 | `score_xp_v2_share_se_pos_prior.md` | +0.0195 [+0.0158, +0.0234] |
| Same-season shares | 2026-10-16 | protocol `score_xp_v2_share_history` | no prior-season fill |
| Live α refresh | 2026-10-16 | protocol `score_xp_v2_alphas` | prior + live counts |
| Live pots (Odds API) | 2026-10-16 | protocol `score_xp_v2_live_props` + quota rule | AH+OU+BTTS; props default **off** |
| Props formula | 2026-10-20 | `score_xp_v2_live_props` | **π from props, level from M**; assists no overlay; lift-only parked |

Assembler: `src/markets/numagent_xp.py`. Defaults: `share_se_shrink=True`, `share_se_prior=position_minutes`, props **off**.

**Match layer (decomposition):** `match_outcome_probs` / `match_outcome_roll` — `src/markets/match_outcome.py` · [`MATCH_OUTCOME.md`](MATCH_OUTCOME.md).

## Strategiser (Numagent 2.0)

| Lock | Date | Report | Note |
| --- | --- | --- | --- |
| L2 μ_post ≡ score_xp_v2 | 2026-10-16 | `se_credibility.md` (historical constant screen) | `mode=none` |
| L3 transfers | 2026-10-14 | `strategy_layer3_audit.md` | audit 6/6 |
| L4 horizon | 2026-10-15 | `strategy_layer4_audit.md` | H=1 ≡ L3 |
| **L4 μ_post priced γ=1** | 2026-10-15/16 | `layer4_mupost_gamma1.md` | **default**; BOOKIE_DEPTH=2 |
| Planning glue (captain=μ) | 2026-10-15 | `fixture_planning_glue.md` | haul/CVaR cover 0 |
| L5 beam K=8 | 2026-10-16 | `layer5_vs_l4_k_rescreen.md` | width only; L4 stays default |

Roadmap: `strategy_layers.md`.

## Parked screens (clear lose or cover 0)

| Arm | Report |
| --- | --- |
| Share-SE all-minutes | `score_xp_v2_share_se_shrink.md` (superseded; DEF/GK leak) |
| Share-SE scorers-only | `score_xp_v2_share_se_shrink_scorers.md` |
| Minutes decay ρ=0.7 | `score_xp_v2_minutes_decay.md` |
| Anytime props as live default | pot-cap distorted Chelsea shares (GW7); re-screen under row-1 before promoting |
| Lift-only / hard absolute λ replace | disagree with scoreline rolls from M; superseded by props-shape |
| L4 cov-as-μ bookie | `layer4_bookie_trial.md` |
| L4 risk / ±SE / window / chase / posture / beyond form / E | `layer4_mupost_*.md` |
| L4 FT-DP, chip half-stop | `layer4_ft_dp.md`, `layer4_chip_stop.md` |
| L5 as strategiser | `layer5_vs_l4.md`, `layer5_vs_l4_rescreen.md` |

## Cascade surfaces

Contract: [`CASCADE_CONTRACT.md`](CASCADE_CONTRACT.md). Plan: [`CASCADE_PLAN.md`](CASCADE_PLAN.md). Scorer checklist: [`SCORE_XP_V2_CHECKLIST.md`](SCORE_XP_V2_CHECKLIST.md).

| | |
| --- | --- |
| Branches | `cascade_short` (priced t:t+1) · `cascade_long` (Opta t+2…) |
| Shared buy universe | `src/models/player_pool.py` (FFScout ∩ minutes) |
| L4 μ inject | `src/strategy/layer4_mu_inject.py` |
| Long λ→xP | `src/markets/cascade_score_v2.py` |

### cascade_short generative (locked 2026-10-20)

Protocol: `cascade_short_generative`. Entry: `src/models/cascade_short.py` → `short_horizon_cascade`.

| Layer | Method |
| --- | --- |
| Mean / independent E[V] | `score_team_side` (moment xCS/xGC) |
| Scenario strands (default on) | λ ±15% → `k ∼ M` → lineup cov → attack dice → CS/GC from rolled GA |
| Soft priors | DefCon `P(hit)`, bonus/cards own-rate, GKP saves multinomial — via `history_priors_for_xp` / same helpers as `score_xp_v2` |
| Deferred | firm XI / news override; bonus-from-roll |

Modules: `short_horizon_generative`, `match_outcome`, `lineup_roll`, `attack_roll`, `defence_roll`, `player_attack`, `prop_shares`.

### cascade_long match outcome (locked 2026-10-20)

Protocol: `cascade_long_match_outcome`. Entry: `cascade_long` → `multistage_mc_milp`.

| Step | Method |
| --- | --- |
| Rank → GF/GA | Locked OLS on closed-season place history (`opta_gf_ga_horizon`) |
| λ | `match_prediction` bands; 0.7 Opta + 0.3 ClubPath blend |
| M / scoreline | `match_outcome_probs_from_lambdas` → `match_outcome_roll` (`k ∼ M`) |
| Player xP | Mean `cascade_score_v2` / `score_team_side` (generative attack deferred) |

## Superseded

- **Numagent 1.0** — `protocol.numagent_1_0` / `SUPERSEDED.md` / **[`ARCHIVE_NUMAGENT_1_0.md`](ARCHIVE_NUMAGENT_1_0.md)** (in-place keep-list)
- **`score_xp`** — historical climbs and gate baseline only (same archive index)
