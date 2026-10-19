# Locked & promoted — Numagent 2.0

Front door for what won. Everything else in `reports/` is evidence, screen history, or parked arms. Do not treat stage_* / GW1–5 / stragent trial notes as the live path.

## Live decision

| Artifact | Path |
| --- | --- |
| **GW7 PI lock** | `data/predictions/2026-27/gw07/numagent_lock.json` |
| Pick report | `reports/numagent_live_pick_gw07.md` |
| Live entrypoint | `python -m src.live.numagent_live_pick --gw N` |

## Scorer (`score_xp_v2` — engine primary)

**`score_xp` superseded 2026-10-19** (`protocol.primary_scorer`). Gate that justified the promotion: vs `score_xp` Δ **+0.0531** [+0.0453, +0.0609].

| Lock | Date | Report | Δ / note |
| --- | --- | --- | --- |
| Primary scorer | 2026-10-13 · primary 2026-10-19 | `score_xp_v2.md`, `numagent_2_0_method.md`, `score_xp_v2_gate.md` | replaces `score_xp` |
| RATE_M0=270 | 2026-10-16 | `score_xp_v2_rate_m0.md` | +0.0037 [+0.0006, +0.0071] |
| Share-SE position×minutes | 2026-10-17 | `score_xp_v2_share_se_pos_prior.md` | +0.0195 [+0.0158, +0.0234] |
| Same-season shares | 2026-10-16 | protocol `score_xp_v2_share_history` | no prior-season fill |
| Live α refresh | 2026-10-16 | protocol `score_xp_v2_alphas` | prior + live counts |

Assembler: `src/markets/numagent_xp.py`. Defaults: `share_se_shrink=True`, `share_se_prior=position_minutes`, props **off**.

## Strategiser

| Lock | Date | Report | Note |
| --- | --- | --- | --- |
| L2 μ_post ≡ score_xp_v2 | 2026-10-16 | `se_credibility.md` (historical constant screen) | `mode=none` |
| L3 transfers | 2026-10-14 | `strategy_layer3_audit.md` | audit 6/6 |
| L4 horizon | 2026-10-15 | `strategy_layer4_audit.md` | H=1 ≡ L3 |
| **L4 μ_post priced γ=1** | 2026-10-15/16 | `layer4_mupost_gamma1.md` | **default**; BOOKIE_DEPTH=2 |
| Planning glue (captain=μ) | 2026-10-15 | `fixture_planning_glue.md` | haul/CVaR cover 0 |
| L5 beam K=8 | 2026-10-16 | `layer5_vs_l4_k_rescreen.md` | width only; L4 stays default |

Roadmap (status table kept current): `strategy_layers.md`.

## Parked screens (clear lose or cover 0)

| Arm | Report |
| --- | --- |
| Share-SE all-minutes | `score_xp_v2_share_se_shrink.md` (superseded; DEF/GK leak) |
| Share-SE scorers-only | `score_xp_v2_share_se_shrink_scorers.md` |
| Minutes decay ρ=0.7 | `score_xp_v2_minutes_decay.md` |
| Anytime props as live default | pot-cap distorted Chelsea shares (GW7) |
| L4 cov-as-μ bookie | `layer4_bookie_trial.md` |
| L4 risk / ±SE / window / chase / posture / beyond form / E | `layer4_mupost_*.md` |
| L4 FT-DP, chip half-stop | `layer4_ft_dp.md`, `layer4_chip_stop.md` |
| L5 as strategiser | `layer5_vs_l4.md`, `layer5_vs_l4_rescreen.md` |

## Superseded scorer

`score_xp` — historical climbs and the gate baseline only. See `SUPERSEDED.md`. Do not use for new live work or new screens except as a named baseline against `score_xp_v2`.
