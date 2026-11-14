# Baudot fixture dots (+OU / +AH line) vs locked gate

Card information trial. Same mixer, learnable Tversky, full-remainder one-shot.
No Odds API — football-data opening OU 2.5 and AHh on closed seasons.

- **ou_only:** 1X2₁₂ + OU₈ thermometer (P(over) left-fill).
- **ou_ah:** (1) + AH₈ **line** bar (assessed AHh). Not AH prices (those track 1X2).

Gate: `baudot_row_mixer_learnable_tversky` · mean 0.218.

## AH redundancy check (fill correlation on ou_ah cards)

- corr(AH fill, 1X2 left-fill) = **0.208**
- corr(OU fill, 1X2 left-fill) = **0.159**
- corr(AH fill, OU fill) = **0.508**

| arm | params | epochs | mean score | Δ gate | seconds |
|---|---:|---:|---:|---:|---:|
| gate (1X2 only) | — | — | 0.218 | +0.000 | — |
| 1_ou_only | 40999 | 20 | 0.176 | -0.042 | 41.5 |
| 2_ou_ah | 43735 | 25 | 0.152 | -0.066 | 51.1 |

## Goals at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| gate_autopsy | 0/0/0 | 103/0/0 | 394/370/83 | 233/331/111 |
| 1_ou_only | 0/0/0 | 103/0/0 | 394/343/67 | 233/260/90 |
| 2_ou_ah | 0/0/0 | 103/1/0 | 394/546/108 | 233/371/118 |

## Assists at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| gate_autopsy | 4/0/0 | 181/0/0 | 434/363/65 | 88/330/49 |
| 1_ou_only | 4/0/0 | 181/0/0 | 434/340/64 | 88/259/39 |
| 2_ou_ah | 4/0/0 | 181/1/0 | 434/552/100 | 88/371/52 |

## Holdout table

| arm | revealed | channel | actual | painted | hits | actual ±1 | painted ±1 | score | Δ gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gate | 20% | goals | 730 | 701 | 194 | 268 | 377 | 0.260 | +0.000 |
| gate | 20% | assists | 707 | 693 | 114 | 153 | 232 | 0.160 | +0.000 |
| gate | 40% | goals | 544 | 557 | 151 | 214 | 299 | 0.268 | +0.000 |
| gate | 40% | assists | 531 | 551 | 89 | 123 | 174 | 0.159 | +0.000 |
| gate | 60% | goals | 360 | 383 | 105 | 145 | 205 | 0.266 | +0.000 |
| gate | 60% | assists | 351 | 377 | 66 | 90 | 126 | 0.169 | +0.000 |
| gate | 80% | goals | 199 | 195 | 60 | 82 | 101 | 0.298 | +0.000 |
| gate | 80% | assists | 192 | 191 | 32 | 45 | 59 | 0.166 | +0.000 |
| 1_ou_only | 20% | goals | 730 | 603 | 157 | 215 | 312 | 0.195 | -0.066 |
| 1_ou_only | 20% | assists | 707 | 599 | 103 | 140 | 219 | 0.134 | -0.026 |
| 1_ou_only | 40% | goals | 544 | 619 | 145 | 208 | 316 | 0.219 | -0.049 |
| 1_ou_only | 40% | assists | 531 | 608 | 100 | 134 | 202 | 0.153 | -0.005 |
| 1_ou_only | 60% | goals | 360 | 472 | 107 | 145 | 230 | 0.196 | -0.069 |
| 1_ou_only | 60% | assists | 351 | 469 | 89 | 105 | 166 | 0.162 | -0.006 |
| 1_ou_only | 80% | goals | 199 | 274 | 67 | 89 | 130 | 0.206 | -0.093 |
| 1_ou_only | 80% | assists | 192 | 273 | 48 | 61 | 90 | 0.145 | -0.021 |
| 2_ou_ah | 20% | goals | 730 | 918 | 226 | 316 | 452 | 0.218 | -0.042 |
| 2_ou_ah | 20% | assists | 707 | 924 | 152 | 202 | 311 | 0.143 | -0.017 |
| 2_ou_ah | 40% | goals | 544 | 892 | 208 | 284 | 419 | 0.177 | -0.091 |
| 2_ou_ah | 40% | assists | 531 | 893 | 141 | 187 | 279 | 0.118 | -0.041 |
| 2_ou_ah | 60% | goals | 360 | 671 | 158 | 209 | 313 | 0.164 | -0.101 |
| 2_ou_ah | 60% | assists | 351 | 673 | 115 | 143 | 216 | 0.117 | -0.052 |
| 2_ou_ah | 80% | goals | 199 | 380 | 95 | 123 | 174 | 0.172 | -0.127 |
| 2_ou_ah | 80% | assists | 192 | 380 | 63 | 85 | 118 | 0.111 | -0.055 |

Primary call: does DEF leave 0/0 on goals and assists, and does assist-by-pos improve without collapsing mean score? AH is judged useful only if ou_ah beats ou_only on those diagnostics after the redundancy check.
