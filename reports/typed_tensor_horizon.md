# Typed tensor — horizon FOV painter

One-shot remainder on the cemented typed tensor. Past true; blank G/A from the cut;
header/minutes stay; log + Brier on every live remainder cell; seats shuffled.
Same mixer as stepped. Train 2022-23…2024-25; holdout 2025-26.

Horizon epochs: 19 (10.2s). Best remainder log: 0.1480.
Stepped (retrain, early-stop on d1): best d1 log 0.1696.

## Remainder window (horizon primary)

| arm | remainder log | remainder Brier | n |
| --- | ---: | ---: | ---: |
| horizon | 0.1480 | 0.0395 | 80234 |

## Holdout by weeks ahead

| arm | weeks ahead | log loss | Brier | on solid | on blank | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| horizon | 1 | 0.1686 | 0.0459 | 0.101 | 0.047 | 4148 |
| stepped | 1 | 0.1696 | 0.0460 | 0.109 | 0.052 | 4148 |
| horizon | 2 | 0.1316 | 0.0351 | 0.107 | 0.045 | 4168 |
| stepped | 2 | 0.1355 | 0.0356 | 0.117 | 0.053 | 4168 |
| horizon | 4 | 0.1458 | 0.0379 | 0.089 | 0.047 | 4182 |
| stepped | 4 | 0.1475 | 0.0382 | 0.103 | 0.054 | 4182 |
| horizon | 8 | 0.1627 | 0.0437 | 0.104 | 0.046 | 4206 |
| stepped | 8 | 0.1637 | 0.0439 | 0.115 | 0.052 | 4206 |

## Spine logistic reference

log loss **0.1497**, Brier **0.0400**.

## Call

Horizon remainder log **0.1480** is only a hair under sliced logistic **0.1497** (Δ=-0.0017) — not a clear win on a different cell set. d1 **0.1686** still trails the spine (stepped d1 **0.1696**). Park as inconclusive; card stays cemented.
