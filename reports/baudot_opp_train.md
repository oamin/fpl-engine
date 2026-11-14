# Baudot opponent-card training

The painted club card keeps its own bits. Each GW row attends to the opponent card: earlier rows true, this fixture's minutes revealed, this fixture's G/A and gc blank, later kickoffs hidden. Minutes are not copied onto the painted card. gc6 sits in the header, so this arm changes the header and the attention together. Loss is the club's remainder events.

Gate mean **0.218**. This arm mean **0.192** (Δ -0.026).

| arm | epochs | α | β | mean | Δ gate | seconds |
|---|---:|---:|---:|---:|---:|---:|
| gate | — | 0.447 | 0.553 | 0.218 | +0.000 | — |
| opp_cross | 20 | 0.447 | 0.553 | 0.192 | -0.026 | 358.8 |

## Holdout table

| revealed | channel | actual | painted | hits | score |
|---|---|---:|---:|---:|---:|
| 20% | goals | 730 | 770 | 197 | 0.249 |
| 20% | assists | 707 | 767 | 131 | 0.164 |
| 40% | goals | 544 | 644 | 155 | 0.220 |
| 40% | assists | 531 | 640 | 108 | 0.153 |
| 60% | goals | 360 | 451 | 106 | 0.209 |
| 60% | assists | 351 | 449 | 84 | 0.164 |
| 80% | goals | 199 | 238 | 57 | 0.218 |
| 80% | assists | 192 | 233 | 41 | 0.159 |

## 20% by position (actual / painted / hits · precision · recall)

### goals

| pos | a/p/h | precision | recall |
|---|---|---:|---:|
| GKP | 0/0/0 | — | — |
| DEF | 103/0/0 | — | 0.000 |
| MID | 394/447/96 | 0.215 | 0.244 |
| FWD | 233/323/101 | 0.313 | 0.433 |

### assists

| pos | a/p/h | precision | recall |
|---|---|---:|---:|
| GKP | 4/0/0 | — | 0.000 |
| DEF | 181/0/0 | — | 0.000 |
| MID | 434/447/86 | 0.192 | 0.198 |
| FWD | 88/320/45 | 0.141 | 0.511 |

## Call

Mean **0.192** vs gate **0.218** (Δ -0.026). 20% goals 730/770/197 (P 0.256, R 0.270) against the gate 730/701/194 (P 0.277, R 0.266). Assists 707/767/131 (P 0.171, R 0.185) against 707/693/114 (P 0.164, R 0.161). Extra hits sit inside extra bars, and the longer remainders overpaint.

DEF goals stay **103/0/0** and DEF assists **181/0/0**. Revealed opponent minutes and past gc do not move those rows.

This arm adds the gc6 header and the opponent cross-read together. The pair does not beat the locked mixer.
