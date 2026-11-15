# Flashcard git cleanup (focused)

**Date:** 2026-11-15 (Europe/London).

## Decision

Product on main stays **Numagent 2.0**. Flashcard / Baudot work is a separate research track.
The **typed tensor card** (`src/models/tensor_card.py`, [`TYPED_TENSOR_CARD.md`](TYPED_TENSOR_CARD.md)) is the source of truth going forward.

## What this cleanup does

- Puts the cemented schema + sliced baselines on a single branch into main (`cursor/typed-tensor-card-c985`).
- Parks Baudot **binary** training: bits / Dice–overlap / Baudot planes are not the card.
- Does **not** mass-close the historical Baudot PR stack. Those branches remain archaeological evidence; do not merge them onto main as the live path.

## What stays parked / retired

| Item | Status |
| --- | --- |
| Baudot binary planes as training targets | retired |
| Hand-built LineupNet / FormColumn means | rejected |
| Reservoir / ZIP / SMOTE / SAE add-ons for this painter | parked |
| Captain / chip simulators | deferred until a pre-registered likelihood ask |
| 2026/27 live holdout fit | frozen — do not touch |

## What to open next (one comparison at a time)

1. Full-card stepped FOV painter on the typed tensor (opacity = P(event); log / Brier).
2. Horizon remainder FOV on the same mixer (sanity after stepped).
3. Survival framing only as a secondary experiment after event probability is honest.

## Note on the PR forest

Dozens of `cursor/baudot-*-c985` / flashcard trial PRs exist off main. Consolidation means **stop extending them** and route new work through the typed tensor module — not a force-close of every historical PR in one turn.
