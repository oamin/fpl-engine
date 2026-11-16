# Typed tensor — zero-gated Poisson exposure GLM

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

## Comparison

| Arm | Spec |
| --- | --- |
| **Control** | `SliceLogistic` on `asof_inputs(..., minutes_mode="actual", include_mates=False)` / `FEATURE_NAMES_LAG1`. |
| **Candidate** | Zero-gated Poisson exposure GLM: `m≤0 → p=1e-6`; `m>0` Poisson on unclipped counts with `log(m/90)` exposure; opacity `1−exp(−λ)`. Features = logistic set without `minutes`. |

## Protocol

- Leave-one-season-out on closed seasons 2022-23…2025-26. `2026-27` untouched.
- Primary: paired-by-GW mean binary opacity log-loss, `Δ = poisson_glm_minus_slice_logistic` via `cluster_interval` (n_boot=1000, seed=0).
- Promote only if 95% upper bound of Δ is strictly &lt; 0 **and** the same holds on the `minutes > 0` subset.

## Decision

**keep_slice_logistic**

Interval Δ log-loss (poisson − logistic): mean +0.00508, 95% [+0.00448, +0.00570].
Active minutes subset: mean +0.01407, 95% [+0.01300, +0.01516].

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Mates off asserted. Product as-of audit passed=True.

## Fold holdouts (pooled log loss)

| holdout | n_test | n_active | log LL | poi LL | Δ mean GW | active Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 11048 | 0.1441 | 0.1502 | +0.00600 | +0.01552 |
| 2023-24 | 19994 | 11082 | 0.1537 | 0.1588 | +0.00488 | +0.01456 |
| 2024-25 | 19649 | 11259 | 0.1490 | 0.1537 | +0.00468 | +0.01305 |
| 2025-26 | 19127 | 11193 | 0.1497 | 0.1545 | +0.00480 | +0.01318 |

## Per-channel (pooled across LOSO predictions)

| arm | channel | scope | log loss | Brier | cal slope | base rate | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| logistic | goal | all | 0.1478 | 0.0406 | 0.998 | 0.0489 | 77878 |
| logistic | assist | all | 0.1506 | 0.0401 | 0.994 | 0.0455 | 77878 |
| logistic | pooled | all | 0.1492 | 0.0403 | 0.996 | 0.0472 | 155756 |
| poisson | goal | all | 0.1562 | 0.0425 | 1.115 | 0.0489 | 77878 |
| poisson | assist | all | 0.1525 | 0.0408 | 0.852 | 0.0455 | 77878 |
| poisson | pooled | all | 0.1543 | 0.0416 | 0.991 | 0.0472 | 155756 |
| logistic | goal | active | 0.2528 | 0.0708 | 0.822 | 0.0854 | 44582 |
| logistic | assist | active | 0.2580 | 0.0701 | 0.746 | 0.0795 | 44582 |
| logistic | pooled | active | 0.2554 | 0.0704 | 0.793 | 0.0824 | 89164 |
| poisson | goal | active | 0.2728 | 0.0742 | 1.116 | 0.0854 | 44582 |
| poisson | assist | active | 0.2664 | 0.0713 | 0.847 | 0.0795 | 44582 |
| poisson | pooled | active | 0.2696 | 0.0728 | 0.994 | 0.0824 | 89164 |

## Call

**keep_slice_logistic.** Poisson GLM loses on the GW-clustered interval (Δ mean +0.00508, entirely above zero). The active-minutes subset is worse still (Δ +0.014), so the loss is not an artefact of zero-gating blanks. Park the exposure GLM; slice spine stays logistic. This screen held mates off for both arms (Co-PI leak fix); mates on vs off is not decided here.

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_poisson_glm`.
