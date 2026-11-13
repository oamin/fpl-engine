# Baudot loss replacements vs locked Tversky gate

Morning table. Same card, mixer, FOV, seed. Only the loss changes.

Gate: `baudot_row_mixer_learnable_tversky` · α=0.447 β=0.553 · mean **0.218**.

| arm | loss | params | epochs | mean score | Δ gate | seconds |
|---|---|---:|---:|---:|---:|---:|
| gate | learnable Tversky β≥α | — | — | 0.218 | +0.000 | — |
| 1_listwise_bal | listwise_bal | 38263 | 18 | 0.181 | -0.037 | 173.3 |
| 2_calib_theta | calib_theta | 38265 | 15 | 0.000 | -0.218 | 431.4 |

## Goals at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| gate (autopsy) | 0/0/0 | 103/0/0 | 394/~370/83 | 233/~331/111 |
| 1_listwise_bal | 0/0/0 | 103/0/0 | 394/688/130 | 233/296/108 |
| 2_calib_theta | 0/0/0 | 103/0/0 | 394/0/0 | 233/0/0 |

## Assists at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| gate (autopsy) | 4/0/0 | 181/0/0 | 434/~363/65 | 88/~330/49 |
| 1_listwise_bal | 4/0/0 | 181/0/0 | 434/568/110 | 88/266/36 |
| 2_calib_theta | 4/0/0 | 181/0/0 | 434/0/0 | 88/0/0 |

## Holdout table (familiar metrics)

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
| 1_listwise_bal | 20% | goals | 730 | 984 | 238 | 349 | 453 | 0.206 | -0.054 |
| 1_listwise_bal | 20% | assists | 707 | 834 | 146 | 223 | 296 | 0.161 | +0.001 |
| 1_listwise_bal | 40% | goals | 544 | 774 | 182 | 270 | 352 | 0.194 | -0.074 |
| 1_listwise_bal | 40% | assists | 531 | 660 | 111 | 170 | 220 | 0.150 | -0.009 |
| 1_listwise_bal | 60% | goals | 360 | 522 | 123 | 181 | 242 | 0.192 | -0.073 |
| 1_listwise_bal | 60% | assists | 351 | 442 | 80 | 119 | 149 | 0.160 | -0.009 |
| 1_listwise_bal | 80% | goals | 199 | 280 | 70 | 101 | 129 | 0.208 | -0.091 |
| 1_listwise_bal | 80% | assists | 192 | 228 | 45 | 65 | 76 | 0.180 | +0.014 |
| 2_calib_theta | 20% | goals | 730 | 0 | 0 | 0 | 0 | 0.000 | -0.260 |
| 2_calib_theta | 20% | assists | 707 | 0 | 0 | 0 | 0 | 0.000 | -0.160 |
| 2_calib_theta | 40% | goals | 544 | 0 | 0 | 0 | 0 | 0.000 | -0.268 |
| 2_calib_theta | 40% | assists | 531 | 0 | 0 | 0 | 0 | 0.000 | -0.159 |
| 2_calib_theta | 60% | goals | 360 | 0 | 0 | 0 | 0 | 0.000 | -0.266 |
| 2_calib_theta | 60% | assists | 351 | 0 | 0 | 0 | 0 | 0.000 | -0.169 |
| 2_calib_theta | 80% | goals | 199 | 0 | 0 | 0 | 0 | 0.000 | -0.298 |
| 2_calib_theta | 80% | assists | 192 | 0 | 0 | 0 | 0 | 0.000 | -0.166 |

### 2_calib_theta learned thresholds

- θ_G = 1.993 · θ_A = 1.993

## How to read in the morning

1. Did DEF goals/assists leave `0/0` with non-trivial hits (not just volume)?
2. Did FWD assist overpaint drop vs gate autopsy (~330 painted / 88 actual)?
3. Did mean score hold near 0.218 without GKP noise?
4. Prefer an arm only if (1)–(3) improve together; volume-only DEF paint is a fail.
