# Flashcard research — cement lock (2026-11-15)

**Status:** locked for further research. **Not** shipped to a Numagent comparison yet.
**Product on main:** still Numagent 2.0 (`protocol.numagent_2_0`).

This file freezes what is true after the typed-tensor batch. New flashcard training
continues from here on a research branch; do not reopen Baudot bits or mix slice into horizon.

## Source of truth

| Piece | Path |
| --- | --- |
| Card schema | `src/models/tensor_card.py` |
| Design note | [`TYPED_TENSOR_CARD.md`](TYPED_TENSOR_CARD.md) |
| This cement | `FLASHCARD_CEMENT.md` |
| Git cleanup | [`FLASHCARD_GIT_CLEANUP.md`](FLASHCARD_GIT_CLEANUP.md) |
| Index | [`LOCKED.md`](LOCKED.md) (research track section) |

The flashcard is a **typed club-season tensor**: raw minutes, unclipped uint G/A,
header (club / opp / home / assessed 1X2). Opacity is **display only**
\(P(\text{count}\ge 1)\). Baudot binary planes are **retired**.

## Two arms (no mix)

| Arm | Role | Default |
| --- | --- | --- |
| **Slice** | Player–GW likelihood spine | Actual scored-week minutes (+ mate minutes). Logistic baseline. |
| **Horizon** | Full-card FOV exam / picture | Past true G/A; blank events from cut; **true minutes stay on the card**. |

Do **not** blend slice and horizon into one predictor. They answer different questions.

## Screens already run (closed seasons; holdout 2025-26)

| Screen | Report | Call |
| --- | --- | --- |
| Slice logistic vs constant | `typed_tensor_card_baselines.md` | Logistic wins (goal 0.1456 vs 0.1886) |
| Stepped FOV painter | `typed_tensor_stepped.md` | **Lost** vs slice (d1 log 0.1689 vs 0.1497) |
| Horizon FOV painter | `typed_tensor_horizon.md` | **Inconclusive** hairline on remainder; keep as FOV arm |
| Slice × horizon mix | `typed_tensor_slice_horizon_mix.md` | **Inconclusive**; **rejected as product path** (no mix) |
| Lag-1 minutes on slice | `typed_tensor_lag1_minutes.md` | **Lost** (pooled Δlog +0.012); honesty reference only |

### Renders

- Stepped d1: `typed_tensor_stepped_d1_arsenal_2025-26.png`
- Horizon: `typed_tensor_horizon_arsenal_2025-26.png`
- Mix (parked evidence): `typed_tensor_slice_horizon_mix_arsenal_2025-26.png`
- Truth prototype: `typed_tensor_card_prototype_arsenal_2025-26.png`

## Modules

| Module | Purpose |
| --- | --- |
| `tensor_card.py` | Schema, loaders, as-of / lag-1 helpers, FOV blank |
| `tensor_card_baselines.py` | Slice logistic vs constant |
| `tensor_card_stepped.py` | Stepped FOV mixer (parked default) |
| `tensor_card_horizon.py` | Horizon FOV mixer |
| `tensor_card_render.py` | Card PNGs |
| `tensor_card_slice_horizon_mix.py` | Parked mix evidence (do not promote) |
| `tensor_card_lag1_minutes.py` | Lag-1 honesty screen |

Tests under `tests/test_tensor_card*.py`.

## Explicitly not done (research room)

PI intent: **more flashcard model training** before any Numagent gate.

Open room (one pre-registered comparison at a time):

- Better slice event model than logistic (still on the typed card; as-of clean).
- Better horizon / full-card training **without** mixing into slice.
- Count heads, calibration, role-specific paint, richer as-of (opp history) — only if locked before totals are read.
- Survival framing only as a secondary experiment.

**Out of scope until flashcard likelihood is clearly stronger:** wiring flashcard-xP vs `score_xp_v2`, climbs, chips.

## Git state after this cement

- One consolidation PR into `main` carries the typed-tensor tree + these docs.
- Historical Baudot / pre-tensor flashcard PRs are **closed** as archaeology (see cleanup note).
- Intermediate typed-tensor stack PRs are **closed** as superseded by the consolidation PR.
- Continue research on a **new** branch off `main` after merge (or off this cement tip): `cursor/flashcard-*-c985`.

## Holdout

Live 2026/27 remain frozen. No fit on that season. No Odds API abuse for this track.
