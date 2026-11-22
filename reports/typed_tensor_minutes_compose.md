# Typed tensor — minutes composition `p·f(μ)` vs plug-in `f(p·μ)`

**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.
Product on main remains Numagent 2.0.

Co-PI follow-up after `promote_minutes_head`: the −0.034 gap vs gated lag-1 was
amplified by hard-zeroing non-structural lag-1 blanks. Next lever is the
**probability composition**, not a harder gate.

## Comparison

| Arm | Spec |
| --- | --- |
| **Control (plug-in)** | Locked models; `p = f_event(x(ĥm))` with `ĥm = p_play·μ₊`. |
| **Candidate (compose)** | Same models; `p = p_play · f_event(x(μ₊))` (no hard gate on `p_play`). |
| **Diagnostic** | Ungated lag-1 into `f_event` (no `1e-6` on lag1==0). |

## Protocol

- Same play / minutes|play / rich event models as `typed_tensor_minutes_head`.
- LOSO on 2022-23…2025-26. `2026-27` untouched.
- Primary: `Δ = compose_minus_plugin` via `cluster_interval` (n_boot=1000, seed=0).
- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ -0.001.
- Soft-promote if hi &lt; 0 on both but mean above materiality.

## Decision

**inconclusive_keep_plugin**

Interval Δ log-loss (compose − plugin): mean -0.00177, 95% [-0.00205, -0.00148].
Active (actual m&gt;0) subset: mean +0.00345, 95% [+0.00298, +0.00392].

Clears zero=False. Meets materiality=False.

Gameweeks per season: 2022-23=36, 2023-24=37, 2024-25=37, 2025-26=37.

Product as-of audit passed=True.

## Fold holdouts (pooled log loss)

| holdout | n | plugin | compose | lag1 ungated | lag1 gated | inactive plugin | inactive compose | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 19108 | 0.1566 | 0.1545 | 0.1567 | 0.1901 | 0.0280 | 0.0185 | -0.00202 |
| 2023-24 | 19994 | 0.1657 | 0.1641 | 0.1659 | 0.2032 | 0.0270 | 0.0180 | -0.00142 |
| 2024-25 | 19649 | 0.1596 | 0.1575 | 0.1607 | 0.1911 | 0.0270 | 0.0183 | -0.00205 |
| 2025-26 | 19127 | 0.1608 | 0.1591 | 0.1616 | 0.1952 | 0.0265 | 0.0185 | -0.00161 |

## Per-channel (n-weighted, all rows)

| arm | channel | log loss | Brier | n |
| --- | --- | ---: | ---: | ---: |
| plugin | goal | 0.1592 | 0.0419 | 77878 |
| plugin | assist | 0.1622 | 0.0411 | 77878 |
| plugin | pooled | 0.1607 | 0.0415 | 155756 |
| compose | goal | 0.1574 | 0.0419 | 77878 |
| compose | assist | 0.1604 | 0.0411 | 77878 |
| compose | pooled | 0.1589 | 0.0415 | 155756 |
| lag1_ungated | goal | 0.1598 | 0.0422 | 77878 |
| lag1_ungated | assist | 0.1627 | 0.0413 | 77878 |
| lag1_ungated | pooled | 0.1613 | 0.0418 | 155756 |

## Call

**inconclusive_keep_plugin.** All-row interval clears zero (and materiality), but
**active** interval is strictly positive — compose loses on players who actually
played. Inactive LL improves (~0.027→0.018). Keep plug-in. Ungated lag-1 ≈ plugin
(~0.161); gated lag-1 (~0.19) stays the confounded reference.

Do **not** hard-gate `p_play` with oracle `1e-6`. Ungated lag-1 is the
unconfounded diagnostic baseline for future pre-match screens.

Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_minutes_compose`.
