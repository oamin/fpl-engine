# Typed tensor — market pot / role interactions

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

Cement open-room item: AH + Dixon–Coles `xg_own` + role× interactions on the
rich zero-gated spine (mates off; present-row history).

## Comparison

| Arm | Spec |
| --- | --- |
| **Control** | Rich zero-gated SliceLogistic (`typed_tensor_rich_baseline` candidate features). |
| **Candidate** | Same + `ah_line_own`, Dixon–Coles `xg_own` (FD opening AH+OU → `M`), and role × {`log1p(m)`, `xg_own`, `m0_rG`, `m0_rA`}. |

## Protocol

- LOSO on 2022-23…2025-26. `2026-27` untouched. Mates off.
- Market pots from frozen `data/processed/scoreline_fd_moments_*.csv` (no Odds API).
- Primary: `Δ = market_role_minus_rich` via `cluster_interval` (n_boot=1000, seed=0).
- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ -0.001.
- Soft-promote if hi &lt; 0 on both but mean above materiality.

## Decision

**inconclusive_keep_rich_baseline**

Interval Δ log-loss (market_role − rich): mean -0.00016, 95% [-0.00035, +0.00004].
Active minutes subset: mean -0.00027, 95% [-0.00060, +0.00006].

Clears zero=False. Meets materiality=False.

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Market fixture coverage (present weeks): hit=3040, miss=0 (keys=1520).

Product as-of audit passed=True.

## Fold holdouts (pooled log loss)

| holdout | n_test | n_active | rich LL | market+role LL | Δ mean GW | active Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 11048 | 0.1395 | 0.1394 | -0.00016 | -0.00028 |
| 2023-24 | 19994 | 11082 | 0.1473 | 0.1468 | -0.00047 | -0.00085 |
| 2024-25 | 19649 | 11259 | 0.1433 | 0.1434 | +0.00005 | +0.00013 |
| 2025-26 | 19127 | 11193 | 0.1446 | 0.1446 | -0.00006 | -0.00010 |

## Per-channel (n-weighted across LOSO folds)

| arm | channel | scope | log loss | Brier | cal slope | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| control | goal | all | 0.1421 | 0.0401 | 0.991 | 77878 |
| control | assist | all | 0.1453 | 0.0398 | 0.982 | 77878 |
| control | pooled | all | 0.1437 | 0.0399 | 0.987 | 155756 |
| candidate | goal | all | 0.1418 | 0.0400 | 0.984 | 77878 |
| candidate | assist | all | 0.1453 | 0.0397 | 0.964 | 77878 |
| candidate | pooled | all | 0.1436 | 0.0399 | 0.977 | 155756 |
| control | goal | active | 0.2482 | 0.0700 | 0.991 | 44582 |
| control | assist | active | 0.2539 | 0.0695 | 0.982 | 44582 |
| control | pooled | active | 0.2510 | 0.0698 | 0.987 | 89164 |
| candidate | goal | active | 0.2477 | 0.0699 | 0.985 | 44582 |
| candidate | assist | active | 0.2538 | 0.0694 | 0.963 | 44582 |
| candidate | pooled | active | 0.2508 | 0.0697 | 0.976 | 89164 |

## Call

**inconclusive_keep_rich_baseline.** Interval covers zero. Keep rich zero-gated spine without market/role extras as default.

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_market_role`.
