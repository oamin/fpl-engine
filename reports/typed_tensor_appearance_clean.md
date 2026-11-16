# Typed tensor — appearance-clean history

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

One lever: compute player history / M0 on prior `minutes > 0` weeks only,
vs averaging over all prior present sheet rows (bench dilution).

## Comparison

| Arm | Spec |
| --- | --- |
| **Control** | Rich zero-gated logistic; history over prior **present** rows. |
| **Candidate** | Same gate + rich extras; history / M0 on prior **appearances** only. |

## Protocol

- LOSO on 2022-23…2025-26. `2026-27` untouched. Mates off.
- Primary: `Δ = appearance_clean_minus_present_history` via `cluster_interval` (n_boot=1000, seed=0).
- Promote if hi &lt; 0 on all and active; prefer active mean ≤ -0.001.

## Decision

**inconclusive_keep_present_history**

Interval Δ log-loss (appearance − present): mean +0.00001, 95% [-0.00006, +0.00007].
Active minutes subset: mean +0.00002, 95% [-0.00009, +0.00012].

Clears zero=False. Meets active materiality=False.

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Product as-of audit passed=True.

## Fold holdouts (pooled log loss)

| holdout | n_test | n_active | present LL | appearance LL | Δ mean GW | active Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 11048 | 0.1395 | 0.1394 | -0.00008 | -0.00013 |
| 2023-24 | 19994 | 11082 | 0.1473 | 0.1473 | +0.00008 | +0.00014 |
| 2024-25 | 19649 | 11259 | 0.1433 | 0.1434 | +0.00007 | +0.00012 |
| 2025-26 | 19127 | 11193 | 0.1446 | 0.1446 | -0.00003 | -0.00005 |

## Per-channel (n-weighted across LOSO folds)

| arm | channel | scope | log loss | Brier | cal slope | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| control | goal | all | 0.1421 | 0.0401 | 0.991 | 77878 |
| control | assist | all | 0.1453 | 0.0398 | 0.982 | 77878 |
| control | pooled | all | 0.1437 | 0.0399 | 0.987 | 155756 |
| candidate | goal | all | 0.1421 | 0.0401 | 0.991 | 77878 |
| candidate | assist | all | 0.1453 | 0.0398 | 0.984 | 77878 |
| candidate | pooled | all | 0.1437 | 0.0399 | 0.988 | 155756 |
| control | goal | active | 0.2482 | 0.0700 | 0.991 | 44582 |
| control | assist | active | 0.2539 | 0.0695 | 0.982 | 44582 |
| control | pooled | active | 0.2510 | 0.0698 | 0.987 | 89164 |
| candidate | goal | active | 0.2483 | 0.0701 | 0.992 | 44582 |
| candidate | assist | active | 0.2538 | 0.0695 | 0.983 | 44582 |
| candidate | pooled | active | 0.2511 | 0.0698 | 0.988 | 89164 |

## Call

**inconclusive_keep_present_history.** Interval covers zero. Keep present-row history default.

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_appearance_clean`.
