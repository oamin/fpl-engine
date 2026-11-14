# Residual of the true card and the opacity card

The painted dot keeps the model probability as its gray. The difference card is truth minus that probability. Past rows are revealed, so they cancel and the residual lives on the remainder event bits.

Mean squared residual is the Brier score. Mean negative log likelihood of the same bits is the log score. Both are strictly proper for a probability. The log score is the likelihood: a dark dot on a blank week costs more than the same mistake costs under Brier. Brier is the residual itself, squared and averaged.

Model: 2026-11-13 mixer, seed 0. Thresholded gate mean on this retrain **0.218**.

Arsenal 2025-26 remainder at 20%, 4500 event bits: Brier **0.0309**, log loss **0.1137**, mean absolute residual **0.0450**.

Picture: `reports/baudot_residual_arsenal_2025-26.png`.

## Holdout remainder, all event bits

| revealed | bits | Brier | log loss | mean |residual| |
|---|---:|---:|---:|---:|
| 20% | 94134 | 0.0197 | 0.0794 | 0.0306 |
| 40% | 72822 | 0.0194 | 0.0775 | 0.0303 |
| 60% | 48066 | 0.0194 | 0.0772 | 0.0304 |
| 80% | 25680 | 0.0194 | 0.0781 | 0.0303 |

Lower is better on both scores. This is a reading of the existing mixer, not a new trained comparison.
