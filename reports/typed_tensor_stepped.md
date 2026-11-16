# Typed tensor — stepped FOV painter

Full-card stepped FOV on the cemented typed tensor (`tensor_card.py`).
Past rows keep true G/A; events from the painted week onward are blank.
Header / who / minutes stay. Opacity = P(count ≥ 1). Loss = log + Brier.
Seats shuffled each batch. Train 2022-23…2024-25; holdout 2025-26.

Epochs run: 24. Wall: 27.2s. Best holdout d1 log loss: 0.1689.

## Holdout by weeks ahead

| weeks ahead | log loss | Brier | opacity on solid | opacity on blank | n cells |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1689 | 0.0457 | 0.106 | 0.053 | 4148 |
| 2 | 0.1366 | 0.0355 | 0.108 | 0.053 | 4168 |
| 4 | 0.1477 | 0.0380 | 0.095 | 0.054 | 4182 |
| 8 | 0.1639 | 0.0437 | 0.107 | 0.053 | 4206 |

## Spine logistic (same seasons, sliced) for reference

log loss **0.1497**, Brier **0.0400**, n=38254 (player–GW × 2 channels).

## Call

Stepped full-card d1 log **0.1689** does not beat sliced logistic **0.1497**. Card stays cemented; painter is a screen loss or needs a follow-up (architecture / FOV), not a Baudot revival.
