# Opacity card — log loss and Brier

The default card is the Baudot row card. Event dots keep the probability. The training loss is the residual of those probabilities against the true bits. Log loss is the likelihood. Brier is the mean squared residual. The opponent-card cross-read is not in this run.

Tversky-trained baseline on this card: mean log loss **0.0781**, mean Brier **0.0195**.

| arm | epochs | mean log loss | Δ log | mean Brier | Δ Brier | 0.5 tally | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| tversky | 20 | 0.0781 | +0.0000 | 0.0195 | +0.0000 | 0.218 | — |
| log_loss | 48 | 0.0543 | -0.0238 | 0.0141 | -0.0054 | 0.000 | 77.4 |
| brier | 28 | 0.0581 | -0.0200 | 0.0141 | -0.0053 | 0.000 | 44.6 |

## Holdout residual

| arm | revealed | bits | log loss | Brier | |residual| |
|---|---|---:|---:|---:|---:|
| tversky | 20% | 94134 | 0.0794 | 0.0197 | 0.0306 |
| tversky | 40% | 72822 | 0.0775 | 0.0194 | 0.0303 |
| tversky | 60% | 48066 | 0.0772 | 0.0194 | 0.0304 |
| tversky | 80% | 25680 | 0.0781 | 0.0194 | 0.0303 |
| log_loss | 20% | 94134 | 0.0556 | 0.0145 | 0.0288 |
| log_loss | 40% | 72822 | 0.0535 | 0.0139 | 0.0288 |
| log_loss | 60% | 48066 | 0.0535 | 0.0139 | 0.0294 |
| log_loss | 80% | 25680 | 0.0545 | 0.0142 | 0.0303 |
| brier | 20% | 94134 | 0.0594 | 0.0145 | 0.0322 |
| brier | 40% | 72822 | 0.0574 | 0.0139 | 0.0322 |
| brier | 60% | 48066 | 0.0573 | 0.0139 | 0.0327 |
| brier | 80% | 25680 | 0.0582 | 0.0142 | 0.0333 |

## 0.5 tally (not the selection metric)

### log_loss  mean 0.000

| revealed | channel | actual | painted | hits | score |
|---|---|---:|---:|---:|---:|
| 20% | goals | 730 | 1 | 0 | 0.000 |
| 20% | assists | 707 | 0 | 0 | 0.000 |
| 40% | goals | 544 | 1 | 0 | 0.000 |
| 40% | assists | 531 | 0 | 0 | 0.000 |
| 60% | goals | 360 | 1 | 0 | 0.000 |
| 60% | assists | 351 | 0 | 0 | 0.000 |
| 80% | goals | 199 | 0 | 0 | 0.000 |
| 80% | assists | 192 | 0 | 0 | 0.000 |

### brier  mean 0.000

| revealed | channel | actual | painted | hits | score |
|---|---|---:|---:|---:|---:|
| 20% | goals | 730 | 0 | 0 | 0.000 |
| 20% | assists | 707 | 0 | 0 | 0.000 |
| 40% | goals | 544 | 0 | 0 | 0.000 |
| 40% | assists | 531 | 0 | 0 | 0.000 |
| 60% | goals | 360 | 0 | 0 | 0.000 |
| 60% | assists | 351 | 0 | 0 | 0.000 |
| 80% | goals | 199 | 0 | 0 | 0.000 |
| 80% | assists | 192 | 0 | 0 | 0.000 |

## Call

Log loss is the training loss of the opacity card. Its holdout mean is **0.0543** (Tversky-trained 0.0781). The Brier arm's mean squared residual is **0.0141** (Tversky-trained 0.0195). The 0.5 tally is recorded beside those scores and does not choose the checkpoint.
