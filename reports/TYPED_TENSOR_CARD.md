# This is the card

**Locked design (cemented 2026-11-15).** Full freeze: [`FLASHCARD_CEMENT.md`](FLASHCARD_CEMENT.md).

The flashcard is a typed tensor for one club-season. Opacity is only a *display*
of event probability — not the storage format. Baudot binary bit-planes are
**retired for training**.

## Schema (`src/models/tensor_card.py`)

| Field | Shape | Type | Note |
| --- | --- | --- | --- |
| `header` via `gw`, `home`, `opp_id`, `odds_1x2` | `[T, …]` | ids / bool / float | assessed club-win / draw / opp-win |
| `who` via `player_id`, `position`, `names` | `[P]` | id / role / str | seats padded to 40 |
| `minutes` | `[P, T]` | float | **raw minutes**, not `/90` |
| `goals` | `[P, T]` | uint | **unclipped** event counts |
| `assists` | `[P, T]` | uint | **unclipped** event counts |
| `present` | `[P, T]` | bool | row exists that week |

Paint / model output is **opacity** \(= P(\text{count} \ge 1)\). Numerals on the truth card show the uint counts.

## Two arms (no mix)

1. **Slice** — player–GW likelihood spine. Default: **actual** scored-week minutes,
   **zero-gated logistic** (`m>0` fit; `m≤0 → p≈0`; see `typed_tensor_zero_gate.md`).
   Entry: `tensor_card_zero_gate.py` / `asof_inputs(..., minutes_mode="actual")`.
2. **Horizon** — full-card FOV. Past true G/A; blank events from the cut; **true minutes
   stay visible**. Entry: `tensor_card_horizon.py`.

Stepped FOV exists as a screened (lost) variant. Slice×horizon mixes are parked evidence only.

## As-of fields

- Opponent id + assessed odds in the header.
- Player history means / last / last-5 from weeks `< t`.
- Lag-1 predicted minutes helper exists (`minutes_mode="lag1"`) — **lost** vs actual on
  the spine; keep as honesty reference, not the default.

## Prototype

![Typed tensor card prototype — Arsenal 2025-26](typed_tensor_card_prototype_arsenal_2025-26.png)

## Screen index

| Report | Call |
| --- | --- |
| [`typed_tensor_card_baselines.md`](typed_tensor_card_baselines.md) | Slice logistic beats constant |
| [`typed_tensor_stepped.md`](typed_tensor_stepped.md) | Stepped painter **lost** vs slice |
| [`typed_tensor_horizon.md`](typed_tensor_horizon.md) | Horizon **inconclusive**; keep FOV arm |
| [`typed_tensor_slice_horizon_mix.md`](typed_tensor_slice_horizon_mix.md) | Mix **rejected** |
| [`typed_tensor_lag1_minutes.md`](typed_tensor_lag1_minutes.md) | Lag-1 **lost**; actual minutes stay |
| [`typed_tensor_poisson_glm.md`](typed_tensor_poisson_glm.md) | Poisson exposure GLM **lost**; keep SliceLogistic |
| [`typed_tensor_zero_gate.md`](typed_tensor_zero_gate.md) | Zero-gated logistic **promoted** |

## Research continue (not Numagent yet)

Optimise flashcard **model training** on this cemented card (slice and/or horizon as
separate arms). Only after a clear likelihood win, consider a gated comparison to
`score_xp_v2`. See [`FLASHCARD_CEMENT.md`](FLASHCARD_CEMENT.md).
