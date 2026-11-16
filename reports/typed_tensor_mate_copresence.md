# Typed tensor — mate co-presence on zero-gated spine

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

PI question: does **who shares the pitch** move event rates? This is not the
old aggregate `mate_minutes` arm (near-collinear with own minutes).

## Comparison

| Arm | Spec |
| --- | --- |
| **Control** | Zero-gated SliceLogistic on `FEATURE_NAMES_LAG1` (no mates). |
| **Candidate** | Same gate + co-presence: DEF/MID/FWD pitch-mate counts, as-of sum/max of mate `mean_goals` / `mean_assists` among those who played. |

Conditional-on-XI spine (same-week who-played); mate history weeks `< t` only.

## Protocol

- LOSO on 2022-23…2025-26. `2026-27` untouched.
- Primary: `Δ = mate_copresence_minus_zero_gated` via `cluster_interval` (n_boot=1000, seed=0).
- Promote if 95% upper bound of Δ &lt; 0 on all rows **and** on `minutes > 0`.

## Decision

**promote_mate_copresence**

Interval Δ log-loss (copresence − control): mean -0.00029, 95% [-0.00049, -0.00007].
Active minutes subset: mean -0.00049, 95% [-0.00085, -0.00012].

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Product as-of audit passed=True.

## Co-presence vs own minutes (2025-26 active rows)

Correlation of each co-presence feature with own `minutes` (old `mate_minutes` was ≈ −0.97). Soft |r| here means a different signal.

| feature | corr with own minutes | mean |
| --- | ---: | ---: |
| `mate_n_on` | -0.149 | 14.197 |
| `mate_def_on` | -0.106 | 4.875 |
| `mate_mid_on` | -0.014 | 6.610 |
| `mate_fwd_on` | 0.042 | 1.769 |
| `mate_sum_mean_goals` | -0.039 | 1.039 |
| `mate_sum_mean_assists` | -0.042 | 0.929 |
| `mate_max_mean_goals` | -0.018 | 0.380 |

## Fold holdouts (pooled log loss)

| holdout | n_test | n_active | control LL | copresence LL | Δ mean GW | active Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 11048 | 0.1400 | 0.1396 | -0.00046 | -0.00080 |
| 2023-24 | 19994 | 11082 | 0.1482 | 0.1482 | -0.00001 | -0.00002 |
| 2024-25 | 19649 | 11259 | 0.1442 | 0.1439 | -0.00030 | -0.00050 |
| 2025-26 | 19127 | 11193 | 0.1452 | 0.1448 | -0.00038 | -0.00065 |

## Per-channel (n-weighted across LOSO folds)

| arm | channel | scope | log loss | Brier | cal slope | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| control | goal | all | 0.1430 | 0.0404 | 0.993 | 77878 |
| control | assist | all | 0.1459 | 0.0399 | 0.985 | 77878 |
| control | pooled | all | 0.1445 | 0.0402 | 0.991 | 155756 |
| candidate | goal | all | 0.1425 | 0.0402 | 0.992 | 77878 |
| candidate | assist | all | 0.1458 | 0.0399 | 0.978 | 77878 |
| candidate | pooled | all | 0.1442 | 0.0400 | 0.987 | 155756 |
| control | goal | active | 0.2498 | 0.0705 | 0.995 | 44582 |
| control | assist | active | 0.2549 | 0.0698 | 0.984 | 44582 |
| control | pooled | active | 0.2523 | 0.0701 | 0.991 | 89164 |
| candidate | goal | active | 0.2490 | 0.0702 | 0.993 | 44582 |
| candidate | assist | active | 0.2547 | 0.0697 | 0.977 | 44582 |
| candidate | pooled | active | 0.2518 | 0.0700 | 0.987 | 89164 |

## Call

**promote_mate_copresence.** Who-shared-the-pitch features improve zero-gated opacity log-loss on all and active rows. Add co-presence to the slice spine (still conditional-on-XI / minutes).

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_mate_copresence`.
