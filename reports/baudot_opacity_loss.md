# Opacity card — log loss and Brier

The default card is the Baudot row card. Event dots keep the probability. Log loss and Brier are the residual on every remainder event bit at that cut. Actual events, painted events, hits, the ±1 columns, and the score are the 0.5 tally used on the 2026-11-13 mixer table. The Tversky event counts are that published table.

Mean log loss **0.0781** / Brier **0.0195** on the Tversky reading. Mean score of its eight tally cells is **0.218**.

| arm | revealed | channel | log loss | Brier | actual events | painted events | hits | actual ±1 | painted ±1 | score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tversky | 20% | goals | 0.0794 | 0.0197 | 730 | 701 | 194 | 268 | 377 | 0.260 |
| tversky | 40% | goals | 0.0775 | 0.0194 | 544 | 557 | 151 | 214 | 299 | 0.268 |
| tversky | 60% | goals | 0.0772 | 0.0194 | 360 | 383 | 105 | 145 | 205 | 0.266 |
| tversky | 80% | goals | 0.0781 | 0.0194 | 199 | 195 | 60 | 82 | 101 | 0.298 |
| tversky | 20% | assists | 0.0794 | 0.0197 | 707 | 693 | 114 | 153 | 232 | 0.160 |
| tversky | 40% | assists | 0.0775 | 0.0194 | 531 | 551 | 89 | 123 | 174 | 0.159 |
| tversky | 60% | assists | 0.0772 | 0.0194 | 351 | 377 | 66 | 90 | 126 | 0.169 |
| tversky | 80% | assists | 0.0781 | 0.0194 | 192 | 191 | 32 | 45 | 59 | 0.166 |
| log_loss | 20% | goals | 0.0556 | 0.0145 | 730 | 1 | 0 | 1 | 1 | 0.000 |
| log_loss | 40% | goals | 0.0535 | 0.0139 | 544 | 1 | 0 | 1 | 1 | 0.000 |
| log_loss | 60% | goals | 0.0535 | 0.0139 | 360 | 1 | 0 | 1 | 1 | 0.000 |
| log_loss | 80% | goals | 0.0545 | 0.0142 | 199 | 0 | 0 | 0 | 0 | 0.000 |
| log_loss | 20% | assists | 0.0556 | 0.0145 | 707 | 0 | 0 | 0 | 0 | 0.000 |
| log_loss | 40% | assists | 0.0535 | 0.0139 | 531 | 0 | 0 | 0 | 0 | 0.000 |
| log_loss | 60% | assists | 0.0535 | 0.0139 | 351 | 0 | 0 | 0 | 0 | 0.000 |
| log_loss | 80% | assists | 0.0545 | 0.0142 | 192 | 0 | 0 | 0 | 0 | 0.000 |
| brier | 20% | goals | 0.0594 | 0.0145 | 730 | 0 | 0 | 0 | 0 | 0.000 |
| brier | 40% | goals | 0.0574 | 0.0139 | 544 | 0 | 0 | 0 | 0 | 0.000 |
| brier | 60% | goals | 0.0573 | 0.0139 | 360 | 0 | 0 | 0 | 0 | 0.000 |
| brier | 80% | goals | 0.0582 | 0.0142 | 199 | 0 | 0 | 0 | 0 | 0.000 |
| brier | 20% | assists | 0.0594 | 0.0145 | 707 | 0 | 0 | 0 | 0 | 0.000 |
| brier | 40% | assists | 0.0574 | 0.0139 | 531 | 0 | 0 | 0 | 0 | 0.000 |
| brier | 60% | assists | 0.0573 | 0.0139 | 351 | 0 | 0 | 0 | 0 | 0.000 |
| brier | 80% | assists | 0.0582 | 0.0142 | 192 | 0 | 0 | 0 | 0 | 0.000 |

## Call

Log loss holdout mean **0.0543** (Tversky 0.0781). Brier arm mean squared residual **0.0141** (Tversky 0.0195), with log loss **0.0581**. On the 0.5 tally the log-loss arm scores **0.000** and the Brier arm **0.000**, against the Tversky table **0.218**. Both residual arms paint under the 0.5 line, so painted events and hits stay at zero.
