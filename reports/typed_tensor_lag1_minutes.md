# Slice spine — actual minutes vs lag-1

Pre-registered honesty screen on the typed-tensor **slice** arm.
Horizon FOV unchanged (still shows true minutes on the card).

- **actual:** scored-week realised minutes + same-week mate minutes.
- **lag1:** `m_hat_t = m_(t-1)` (else last present before t, else train role prior); mate features dropped.

Train 2022-23, 2023-24, 2024-25 → test 2025-26. n_train=58751, n_test=19127.
Lag-1 minutes MAE on holdout present rows: **15.31** (n=19127).

## Holdout

| arm | event | log loss | Brier | base rate |
| --- | --- | ---: | ---: | ---: |
| actual | goal | 0.1456 | 0.0395 | 0.0466 |
| lag1 | goal | 0.1568 | 0.0410 | 0.0466 |
| actual | assist | 0.1538 | 0.0405 | 0.0454 |
| lag1 | assist | 0.1666 | 0.0418 | 0.0454 |
| actual | pooled | 0.1497 | 0.0400 | 0.0460 |
| lag1 | pooled | 0.1617 | 0.0414 | 0.0460 |

## Delta (lag1 − actual)

| | Δ log | Δ Brier |
| --- | ---: | ---: |
| goal | +0.0112 | +0.0015 |
| assist | +0.0128 | +0.0013 |
| pooled | +0.0120 | +0.0014 |

## Call

Lag-1 **loses** as expected (pooled Δlog **+0.0120**). Actual minutes stay the slice spine; lag-1 is the honest pre-match reference, not a promote.
