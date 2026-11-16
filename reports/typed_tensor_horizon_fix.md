# Typed tensor — horizon painter fixes vs rich spine

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

Co-PI follow-up after confounded market/role× screen: fix the typed-card mixer
defects, then judge on the same player–GW likelihood protocol as the rich
zero-gated spine.

## Comparison

| Arm | Spec |
| --- | --- |
| **Control** | Rich zero-gated SliceLogistic (mates off; present-row history). |
| **Candidate** | Fixed `TypedCardMixer`: season-scoped player codes, full `present` time-mask, structural `m=0 → p≈1e-6` zero-gate, **pure** binary log-loss, early-stop on train-internal val (never holdout). Paint one week with past true G/A. |

## Protocol

- LOSO on 2022-23…2025-26. `2026-27` untouched.
- Same `iter_slices` player–GW cells as rich baseline (`min_history=1`, actual minutes).
- Primary: `Δ = horizon_fix_minus_rich` via `cluster_interval` (n_boot=1000, seed=0).
- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ -0.001.
- Soft-promote if hi &lt; 0 on both but mean above materiality.

## Decision

**keep_rich_baseline**

Interval Δ log-loss (horizon_fix − rich): mean +0.01404, 95% [+0.01304, +0.01512].
Active minutes subset: mean +0.02451, 95% [+0.02275, +0.02640].

Clears zero=False. Meets materiality=False.

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Product as-of audit passed=True.

## Train meta (per LOSO fold)

| holdout | epochs | best val log | seconds | n_fit | n_val |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19 | 0.1220 | 7.1 | 48 | 12 |
| 2023-24 | 19 | 0.1165 | 8.1 | 48 | 12 |
| 2024-25 | 10 | 0.1333 | 3.9 | 48 | 12 |
| 2025-26 | 12 | 0.1391 | 4.3 | 48 | 12 |

## Fold holdouts (pooled log loss)

| holdout | n_test | n_active | rich LL | fixed-mixer LL | Δ mean GW | active Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 11048 | 0.1395 | 0.1489 | +0.00935 | +0.01620 |
| 2023-24 | 19994 | 11082 | 0.1473 | 0.1584 | +0.01074 | +0.01949 |
| 2024-25 | 19649 | 11259 | 0.1433 | 0.1637 | +0.02048 | +0.03562 |
| 2025-26 | 19127 | 11193 | 0.1446 | 0.1601 | +0.01547 | +0.02649 |

## Per-channel (n-weighted across LOSO folds)

| arm | channel | scope | log loss | Brier | cal slope | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| control | goal | all | 0.1421 | 0.0401 | 0.991 | 77878 |
| control | assist | all | 0.1453 | 0.0398 | 0.982 | 77878 |
| control | pooled | all | 0.1437 | 0.0399 | 0.987 | 155756 |
| candidate | goal | all | 0.1598 | 0.0436 | 5.130 | 77878 |
| candidate | assist | all | 0.1558 | 0.0414 | 2.169 | 77878 |
| candidate | pooled | all | 0.1578 | 0.0425 | 3.561 | 155756 |
| control | goal | active | 0.2482 | 0.0700 | 0.991 | 44582 |
| control | assist | active | 0.2539 | 0.0695 | 0.982 | 44582 |
| control | pooled | active | 0.2510 | 0.0698 | 0.987 | 89164 |
| candidate | goal | active | 0.2792 | 0.0761 | 5.172 | 44582 |
| candidate | assist | active | 0.2721 | 0.0723 | 3.022 | 44582 |
| candidate | pooled | active | 0.2756 | 0.0742 | 3.589 | 89164 |

## Call

**keep_rich_baseline.** Fixed mixer loses. Keep rich zero-gated spine; horizon FOV arm stays exam-only.

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_horizon_fix`.
