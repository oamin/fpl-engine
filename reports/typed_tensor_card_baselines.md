# Typed tensor card — sliced baselines

Pre-registered spine check on the cemented typed card (`src/models/tensor_card.py`).
Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
n_train=58751, n_test=19127, cards=80.

Targets: `y = 1[count ≥ 1]` for goals and assists. Features: as-of history
plus scored-week minutes / home / role / assessed 1X2 / teammate shared minutes.
Baudot binary planes are not used.

## Goal event

| Model | log loss | Brier | cal slope | cal intercept | base rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant | 0.1886 | 0.0445 | — | — | 0.0466 |
| logistic | 0.1456 | 0.0395 | 0.977 | -0.100 | 0.0466 |

## Assist event

| Model | log loss | Brier | cal slope | cal intercept | base rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant | 0.1848 | 0.0434 | — | — | 0.0454 |
| logistic | 0.1538 | 0.0405 | 0.931 | -0.152 | 0.0454 |

## Read

Logistic beats constant prevalence on both events when Δlog_loss < 0.
This is a spine sanity check, not a full-card stepped/horizon comparison.
Next: full-card FOV painter on the same tensor (opacity = P(event)).
