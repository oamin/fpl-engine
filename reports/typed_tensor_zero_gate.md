# Typed tensor — zero-gated SliceLogistic

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

Follow-up to `typed_tensor_poisson_glm.md` (Co-PI diagnosis): gate structural
`minutes == 0` zeros on the logistic spine.

## Comparison

| Arm | Spec |
| --- | --- |
| **Control** | Ungated SliceLogistic on all present rows; `FEATURE_NAMES_LAG1`; mates off. |
| **Candidate** | Zero-gated SliceLogistic: fit only on `minutes > 0`; `minutes ≤ 0 → p = 1e-6`. Same features. |

## Protocol

- Leave-one-season-out on 2022-23…2025-26. `2026-27` untouched.
- Primary: paired-by-GW mean binary log-loss, `Δ = zero_gate_minus_ungated_logistic` via `cluster_interval` (n_boot=1000, seed=0).
- Promote only if 95% upper bound of Δ is strictly &lt; 0 **and** the same holds on the `minutes > 0` subset.

## Decision

**promote_zero_gated_logistic**

Interval Δ log-loss (gated − ungated): mean -0.00473, 95% [-0.00507, -0.00437].
Active minutes subset: mean -0.00310, 95% [-0.00369, -0.00250].

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Mates off asserted. Product as-of audit passed=True.

## Fold holdouts (pooled log loss)

| holdout | n_test | n_active | n_zero | ungated LL | gated LL | Δ mean GW | active Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 11048 | 8060 | 0.1441 | 0.1400 | -0.00406 | -0.00189 |
| 2023-24 | 19994 | 11082 | 8912 | 0.1537 | 0.1482 | -0.00556 | -0.00425 |
| 2024-25 | 19649 | 11259 | 8390 | 0.1490 | 0.1442 | -0.00476 | -0.00341 |
| 2025-26 | 19127 | 11193 | 7934 | 0.1497 | 0.1452 | -0.00453 | -0.00283 |

## Per-channel (n-weighted across LOSO folds)

| arm | channel | scope | log loss | Brier | cal slope | base rate | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ungated | goal | all | 0.1478 | 0.0406 | 0.998 | 0.0489 | 77878 |
| ungated | assist | all | 0.1506 | 0.0401 | 0.994 | 0.0455 | 77878 |
| ungated | pooled | all | 0.1492 | 0.0403 | 0.996 | 0.0472 | 155756 |
| gated | goal | all | 0.1430 | 0.0404 | 0.993 | 0.0489 | 77878 |
| gated | assist | all | 0.1459 | 0.0399 | 0.985 | 0.0455 | 77878 |
| gated | pooled | all | 0.1445 | 0.0402 | 0.991 | 0.0472 | 155756 |
| ungated | goal | active | 0.2528 | 0.0708 | 0.822 | 0.0854 | 44582 |
| ungated | assist | active | 0.2580 | 0.0701 | 0.746 | 0.0795 | 44582 |
| ungated | pooled | active | 0.2554 | 0.0704 | 0.793 | 0.0824 | 89164 |
| gated | goal | active | 0.2498 | 0.0705 | 0.995 | 0.0854 | 44582 |
| gated | assist | active | 0.2549 | 0.0698 | 0.984 | 0.0795 | 44582 |
| gated | pooled | active | 0.2523 | 0.0701 | 0.991 | 0.0824 | 89164 |

## Call

**promote_zero_gated_logistic.** Zero-gate wins on the GW-clustered interval including the active-minutes subset. Adopt gated SliceLogistic as the slice spine default (mates still off in this screen; mates on/off not re-opened here).

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_zero_gate`.
