# Typed tensor — pre-match minutes head vs lag-1 honesty

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

Co-PI / Cursor pivot after fixed-mixer loss: stop oracle-minutes feature search.
Attack the **+0.012** lag-1 honesty gap with a pre-match minutes head feeding the
locked rich zero-gated event spine.

## Comparison

| Arm | Spec |
| --- | --- |
| **Control** | Rich zero-gated event head; **lag-1** minutes at inference. |
| **Candidate** | Same event head; **minutes head** `ĥm = P(m>0)·E[m\|m>0]` (as-of only). |
| **Oracle (diagnostic)** | Same event head with **actual** minutes (ceiling). |

## Protocol

- Event head trained on actual minutes (mates off); inference substitutes minutes.
- Minutes head: logistic play + Ridge minutes|play on as-of features (includes lag-1).
- LOSO on 2022-23…2025-26. `2026-27` untouched. Neural slice arm parked.
- Primary: `Δ = minutes_head_minus_lag1` via `cluster_interval` (n_boot=1000, seed=0).
- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ -0.001.

## Decision

**promote_minutes_head**

Interval Δ log-loss (minutes_head − lag1): mean -0.03408, 95% [-0.03818, -0.03041].
Active (actual m&gt;0) subset: mean -0.07073, 95% [-0.07795, -0.06413].
Oracle − lag1 (diagnostic ceiling): mean -0.05098, 95% [-0.05546, -0.04692].

Clears zero=True. Meets materiality=True.

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Product as-of audit passed=True.

## Fold holdouts

| holdout | n_test | lag1 LL | head LL | oracle LL | Δ head−lag1 | minutes MAE lag1 | MAE head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 0.1901 | 0.1566 | 0.1395 | -0.03314 | 15.59 | 18.73 |
| 2023-24 | 19994 | 0.2032 | 0.1657 | 0.1473 | -0.03732 | 14.76 | 18.42 |
| 2024-25 | 19649 | 0.1911 | 0.1596 | 0.1433 | -0.03122 | 15.66 | 18.49 |
| 2025-26 | 19127 | 0.1952 | 0.1608 | 0.1446 | -0.03461 | 15.31 | 18.70 |

## Per-channel (n-weighted, all rows)

| arm | channel | log loss | Brier | n |
| --- | --- | ---: | ---: | ---: |
| lag1 | goal | 0.1931 | 0.0423 | 77878 |
| lag1 | assist | 0.1968 | 0.0414 | 77878 |
| lag1 | pooled | 0.1950 | 0.0418 | 155756 |
| minutes_head | goal | 0.1592 | 0.0419 | 77878 |
| minutes_head | assist | 0.1622 | 0.0411 | 77878 |
| minutes_head | pooled | 0.1607 | 0.0415 | 155756 |
| oracle | goal | 0.1421 | 0.0401 | 77878 |
| oracle | assist | 0.1453 | 0.0398 | 77878 |
| oracle | pooled | 0.1437 | 0.0399 | 155756 |

## Call

**promote_minutes_head.** Pre-match minutes head beats lag-1 on the rich event spine (clears zero + materiality). Adopt as the pre-match minutes path into the locked rich event head.

### Mechanism note (not a demote)

Minutes **MAE is worse** than lag-1 (~18.5 vs ~15.5) while event log-loss is much better. Diagnosis: lag-1 is exactly 0 on ~41% of present rows → hard zero-gate; the head’s `ĥm = P(play)·E[m|play]` is **never** exactly 0 (soft minutes on benches / returners). Correlation with actual minutes is slightly higher for the head (~0.75 vs ~0.73). The promote is for the **event-likelihood** estimand under substituted minutes + zero-gate, not for MAE. A follow-up could threshold `p_play` into a hard gate if shipping requires sparse zeros.

Neural slice / FOV mixer stays parked as a product path under oracle minutes.

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_minutes_head`.
