# Typed tensor — rich zero-gated baseline

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

Co-PI follow-up after mate co-presence review: soft-park co-presence;
upgrade the single-player zero-gated spine (mates off).

## Comparison

| Arm | Spec |
| --- | --- |
| **Control** | Zero-gated SliceLogistic on `FEATURE_NAMES_LAG1` (mates off). |
| **Candidate** | Same gate + `log1p(m)`, `I(m≥60)`, `I(m=90)`, `p_over25`, team as-of G/A, M0=270 own per-90. |

## Protocol

- LOSO on 2022-23…2025-26. `2026-27` untouched.
- Primary: `Δ = rich_baseline_minus_zero_gated` via `cluster_interval` (n_boot=1000, seed=0).
- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ -0.001 (materiality bar).
- Soft-promote if hi &lt; 0 on both but mean above materiality.

## Cement side-effect

Mate co-presence (`typed_tensor_mate_copresence.md`) is **soft-parked** as exploratory.

## Decision

**soft_promote_rich_baseline**

Interval Δ log-loss (rich − plain): mean -0.00072, 95% [-0.00103, -0.00045].
Active minutes subset: mean -0.00125, 95% [-0.00175, -0.00081].

Clears zero=True. Meets materiality=False.

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Product as-of audit passed=True.

## Fold holdouts (pooled log loss)

| holdout | n_test | n_active | plain LL | rich LL | Δ mean GW | active Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 11048 | 0.1400 | 0.1395 | -0.00049 | -0.00086 |
| 2023-24 | 19994 | 11082 | 0.1482 | 0.1473 | -0.00088 | -0.00161 |
| 2024-25 | 19649 | 11259 | 0.1442 | 0.1433 | -0.00094 | -0.00161 |
| 2025-26 | 19127 | 11193 | 0.1452 | 0.1446 | -0.00056 | -0.00089 |

## Per-channel (n-weighted across LOSO folds)

| arm | channel | scope | log loss | Brier | cal slope | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| control | goal | all | 0.1430 | 0.0404 | 0.993 | 77878 |
| control | assist | all | 0.1459 | 0.0399 | 0.985 | 77878 |
| control | pooled | all | 0.1445 | 0.0402 | 0.991 | 155756 |
| candidate | goal | all | 0.1421 | 0.0401 | 0.991 | 77878 |
| candidate | assist | all | 0.1453 | 0.0398 | 0.982 | 77878 |
| candidate | pooled | all | 0.1437 | 0.0399 | 0.987 | 155756 |
| control | goal | active | 0.2498 | 0.0705 | 0.995 | 44582 |
| control | assist | active | 0.2549 | 0.0698 | 0.984 | 44582 |
| control | pooled | active | 0.2523 | 0.0701 | 0.991 | 89164 |
| candidate | goal | active | 0.2482 | 0.0700 | 0.991 | 44582 |
| candidate | assist | active | 0.2539 | 0.0695 | 0.982 | 44582 |
| candidate | pooled | active | 0.2510 | 0.0698 | 0.987 | 89164 |

## Call

**soft_promote_rich_baseline.** Rich features clear zero but miss the −0.001 materiality bar. Prefer rich features for further slice work; do not treat as a large locked win. Co-presence remains soft-parked.

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_rich_baseline`.
