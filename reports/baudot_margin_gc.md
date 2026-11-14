# Margin pools and both-card gc

Both arms start from the locked opponent-card method. Cell Tversky stays. `1_margins` adds role pools and row/column sums at weight 1. `2_gc_scoreline` adds gc for the assessed club and the opponent at weight 1, distributes that scoreline across the revealed XI, and on away fixtures fills the opponent XI in the same step (weight 1).

Opponent-card baseline **0.192**. 2026-11-13 mixer **0.218**.

| arm | epochs | α | β | mean | Δ opp | Δ mixer | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| opp_cross | 20 | 0.447 | 0.553 | 0.192 | +0.000 | -0.026 | 358.8 |
| 1_margins | 15 | 0.442 | 0.558 | 0.000 | -0.192 | -0.218 | 272.9 |
| 2_gc_scoreline | 19 | 0.447 | 0.553 | 0.178 | -0.014 | -0.040 | 355.8 |

## 1_margins

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

### 20% by position

**goals**

| pos | a/p/h | precision | recall |
|---|---|---:|---:|
| GKP | 0/0/0 | — | — |
| DEF | 103/0/0 | — | 0.000 |
| MID | 394/0/0 | — | 0.000 |
| FWD | 233/0/0 | — | 0.000 |

**assists**

| pos | a/p/h | precision | recall |
|---|---|---:|---:|
| GKP | 4/0/0 | — | 0.000 |
| DEF | 181/0/0 | — | 0.000 |
| MID | 434/0/0 | — | 0.000 |
| FWD | 88/0/0 | — | 0.000 |

DEF goals 103/0/0. DEF assists 181/0/0. Mean 0.000.


## 2_gc_scoreline

| revealed | channel | actual | painted | hits | score |
|---|---|---:|---:|---:|---:|
| 20% | goals | 730 | 634 | 181 | 0.230 |
| 20% | assists | 707 | 1255 | 185 | 0.106 |
| 40% | goals | 544 | 492 | 142 | 0.248 |
| 40% | assists | 531 | 983 | 146 | 0.104 |
| 60% | goals | 360 | 325 | 94 | 0.248 |
| 60% | assists | 351 | 665 | 103 | 0.107 |
| 80% | goals | 199 | 178 | 57 | 0.270 |
| 80% | assists | 192 | 361 | 57 | 0.110 |

### 20% by position

**goals**

| pos | a/p/h | precision | recall |
|---|---|---:|---:|
| GKP | 0/0/0 | — | — |
| DEF | 103/0/0 | — | 0.000 |
| MID | 394/64/11 | 0.172 | 0.028 |
| FWD | 233/570/170 | 0.298 | 0.730 |

**assists**

| pos | a/p/h | precision | recall |
|---|---|---:|---:|
| GKP | 4/0/0 | — | 0.000 |
| DEF | 181/0/0 | — | 0.000 |
| MID | 434/660/118 | 0.179 | 0.272 |
| FWD | 88/595/67 | 0.113 | 0.761 |

DEF goals 103/0/0. DEF assists 181/0/0. Mean 0.178.

## Call

`1_margins` stays under the 0.5 threshold for every epoch (mean **0.000**). The pool terms match soft counts, and a row total can match with every bit sat under 0.5. Early stop keeps that blank picture.

`2_gc_scoreline` mean **0.178** (Δ -0.014 vs the opponent card, Δ -0.040 vs the 2026-11-13 mixer). 20% goals are 730/634/181 (P 0.285, R 0.248). The share puts those goals on forwards (233/570/170, recall 0.730) and almost none on midfielders (394/64/11, recall 0.028). Assists stay a free head and overpaint (707/1255/185). DEF goals and assists stay **103/0/0** and **181/0/0**.

Neither arm replaces the cell-Tversky opponent card.
