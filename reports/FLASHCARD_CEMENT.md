# Flashcard research — cement lock (2026-11-16)

**Status:** locked. Batch freeze after typed-tensor cement + spine screens.
**Not** shipped to a Numagent / `score_xp_v2` comparison.
**Product on main:** still Numagent 2.0 (`protocol.numagent_2_0`).

This file is the source of truth for what is locked on the flashcard track.
Do not reopen Baudot planes, mix slice into horizon, or fit on live 2026/27.

## Source of truth

| Piece | Path |
| --- | --- |
| Card schema | `src/models/tensor_card.py` |
| Design note | [`TYPED_TENSOR_CARD.md`](TYPED_TENSOR_CARD.md) |
| This cement | `FLASHCARD_CEMENT.md` |
| Git cleanup | [`FLASHCARD_GIT_CLEANUP.md`](FLASHCARD_GIT_CLEANUP.md) |
| Index | [`LOCKED.md`](LOCKED.md) (research track section) |

## Card

Typed club-season tensor: raw minutes, unclipped uint G/A, header (club / opp /
home / assessed 1X2). Opacity is **display only** \(P(\text{count}\ge 1)\).
Baudot binary planes are **retired**.

## Locked spine (slice arm)

| Piece | Lock |
| --- | --- |
| Minutes (oracle / research) | Actual scored-week minutes for conditional event training. Lag-1 **lost** as event minutes. |
| Minutes (pre-match path) | **Two-part minutes head** `ĥm=P(m>0)·E[m\|m>0]` (**promoted** vs lag-1 on event LL). |
| Event head | **Zero-gated SliceLogistic** — fit on `m>0` only; `m≤0 → p≈1e-6`. **Promoted.** |
| Rich extras | Prefer for further work (**soft-promote**): `log1p(m)`, `I(m≥60)`, `I(m=90)`, `p_over25`, team as-of G/A, `M0=270` own per-90. |
| Mates | Aggregate `mate_minutes` off. Mate **co-presence** screened then **soft-parked** (Co-PI: headcount/sub leak, not chemistry). |
| Poisson GLM | Exposure GLM **lost** (α misspec + rigid exposure); do not reopen without a new twist. |
| Neural mixer | Fixed mixer **lost** vs rich under oracle minutes; FOV exam-only. |

## Horizon arm

Full-card FOV: past true G/A; blank events from cut; **true minutes stay**.
Keep as FOV exam arm. Stepped FOV **lost**. Slice×horizon mix **rejected**.
Do **not** blend into the slice predictor.

## Screens (closed seasons; 4-season LOSO where noted)

| Screen | Report | Call |
| --- | --- | --- |
| Slice logistic vs constant | `typed_tensor_card_baselines.md` | Logistic wins (goal 0.1456 vs 0.1886) |
| Stepped FOV painter | `typed_tensor_stepped.md` | **Lost** vs slice (d1 0.1689 vs 0.1497) |
| Horizon FOV painter | `typed_tensor_horizon.md` | **Inconclusive** hairline; keep FOV arm |
| Slice × horizon mix | `typed_tensor_slice_horizon_mix.md` | **Rejected** as product path |
| Lag-1 minutes | `typed_tensor_lag1_minutes.md` | **Lost** (+0.012 pooled Δlog) |
| Poisson exposure GLM | `typed_tensor_poisson_glm.md` | **Lost** (+0.00508 [+0.00448, +0.00570]) |
| Zero-gated SliceLogistic | `typed_tensor_zero_gate.md` | **Promoted** (−0.00473 [−0.00507, −0.00437]) |
| Mate co-presence | `typed_tensor_mate_copresence.md` | Screened then **soft-parked** |
| Rich zero-gated baseline | `typed_tensor_rich_baseline.md` | **Soft-promote** (−0.00072 [−0.00103, −0.00045]) |
| Appearance-clean history | `typed_tensor_appearance_clean.md` | **Inconclusive** (Δ ≈ 0; keep present-row history) |
| Market pot / role× | `typed_tensor_market_role.md` | **Inconclusive + confounded** (venue-less join; keep rich) |
| Horizon painter fixes | `typed_tensor_horizon_fix.md` | **Lost** vs rich (+0.014; keep rich spine) |
| Pre-match minutes head | `typed_tensor_minutes_head.md` | **Promoted** vs lag-1 (Δlog −0.034) |

### Renders

- Stepped d1: `typed_tensor_stepped_d1_arsenal_2025-26.png`
- Horizon: `typed_tensor_horizon_arsenal_2025-26.png`
- Mix (parked): `typed_tensor_slice_horizon_mix_arsenal_2025-26.png`
- Truth prototype: `typed_tensor_card_prototype_arsenal_2025-26.png`

## Modules

| Module | Purpose |
| --- | --- |
| `tensor_card.py` | Schema, loaders, as-of / lag-1 / co-presence helpers, FOV blank |
| `tensor_card_baselines.py` | Slice logistic vs constant |
| `tensor_card_zero_gate.py` | Zero-gated logistic (**promoted**) |
| `tensor_card_rich_baseline.py` | Rich extras on zero-gated spine (**soft-promote**) |
| `tensor_card_appearance_clean.py` | Appearance-clean history (inconclusive; keep present) |
| `tensor_card_market_role.py` | Market pot + role× (inconclusive + confounded; keep rich) |
| `tensor_card_horizon_fix.py` | Fixed mixer LOSO vs rich (**lost**) |
| `tensor_card_minutes_head.py` | Pre-match minutes head vs lag-1 (**promoted**) |
| `tensor_card_poisson_glm.py` | Poisson GLM (lost; parked) |
| `tensor_card_mate_copresence.py` | Co-presence (soft-parked) |
| `tensor_card_horizon.py` | Horizon FOV mixer |
| `tensor_card_stepped.py` | Stepped FOV mixer (parked) |
| `tensor_card_slice_horizon_mix.py` | Mix evidence (rejected) |
| `tensor_card_lag1_minutes.py` | Lag-1 honesty |
| `tensor_card_render.py` | Card PNGs |

Tests: `tests/test_tensor_card*.py`.

## Co-PI lessons locked in

1. **Zero-gate** structural `m=0` blanks on the logistic spine (promoted).
2. **Poisson α=1.0** was a sklearn scaling trap; even after fixes, binary logistic edges Poisson on opacity — parked.
3. **Mate co-presence** as additive `minutes>0` counts is not “teammate chemistry”; soft-park. Joint mate correlation belongs on full-card / horizon, not additive slice logits.
4. **Rich single-player baseline** before further mate or mixer work.
5. **Market/role× screen confounded** by venue-less pot join; not a clean `xg_own` falsification.
6. **Fixed mixer** (season codes + present mask + zero-gate + pure log + val early-stop) still **loses** vs rich on 4-season player–GW LOSO — keep rich spine.
7. **Pre-match minutes head** beats lag-1 on event LL (Δ −0.034) even with worse MAE — soft `ĥm` vs hard lag-1 zeros; promote on likelihood estimand.

## Open room (after this lock)

One pre-registered comparison at a time, from this cement:

- Harden minutes head (e.g. `p_play` hard-gate threshold) if sparse zeros are required for shipping.
- Count / rate expectation + team pot conservation toward a future Numagent bridge (only after pre-match path is solid).
- Do **not** reopen Baudot, Poisson (without new twist), co-presence-as-spine, lag-1 as default minutes, appearance-clean, market/role× (confounded), fixed-mixer-as-slice (lost +0.014), or oracle-minutes feature tweaks (opp history / starter-mask).
- Horizon FOV remains an exam arm only; do not mix into the slice predictor.

**Out of scope until flashcard likelihood clearly earns it:** flashcard-xP vs `score_xp_v2`, climbs, chips.

## Git

Stack that carries this lock (merge bottom-up or land tip on `main`):

| PR | Branch | Role |
| --- | --- | --- |
| #191 | `cursor/flashcard-cement-c985` → `main` | Typed-tensor cement onto main |
| #192 | `…poisson-glm…` | Poisson screen (lost) |
| #194 | `…zero-gate…` | Zero-gate promote |
| #195 | `…mate-copresence…` | Co-presence (later soft-parked) |
| #196 | `…rich-baseline…` | Rich soft-promote + park co-presence |
| consolidation | tip → `main` | This batch lock |

Continue later research on a **new** branch off `main` after the tip lands: `cursor/flashcard-*-c985`.

## Holdout

Live 2026/27 frozen (`data/live/HOLDOUT_FREEZE.json`). No fit on that season.
No Odds API abuse for this track (prefer football-data / snapshots).
