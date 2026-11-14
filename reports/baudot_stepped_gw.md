# Baudot stepped single-GW vs locked full-remainder gate

Complementary FOV. Same card, BlockTokenMixer, learnable Tversky.
- **Gate:** all past true → one-shot paint of the entire remainder window.
- **This arm:** all past true → one-shot paint of the next GW only, stepped;
  holdout is teacher-forced (true G/A on earlier remainder weeks).
Gate method: `baudot_row_mixer_learnable_tversky` · mean score 0.218.

| arm | params | epochs | α | β | mean score | Δ gate | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate (full remainder one-shot) | — | — | 0.447 | 0.553 | 0.218 | +0.000 | — |
| stepped_next_gw | 38263 | 37 | 0.451 | 0.549 | 0.184 | -0.034 | 830.8 |

## Goals at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| stepped_next_gw | 0/0/0 | 103/0/0 | 394/450/95 | 233/457/143 |

## Assists at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| stepped_next_gw | 4/0/0 | 181/1/0 | 434/466/95 | 88/462/50 |

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
| stepped_next_gw | 20% | goals | 730 | 907 | 238 | 309 | 508 | 0.234 | -0.026 |
| stepped_next_gw | 20% | assists | 707 | 929 | 145 | 181 | 306 | 0.135 | -0.025 |
| stepped_next_gw | 40% | goals | 544 | 681 | 176 | 234 | 387 | 0.230 | -0.038 |
| stepped_next_gw | 40% | assists | 531 | 698 | 105 | 134 | 222 | 0.130 | -0.029 |
| stepped_next_gw | 60% | goals | 360 | 452 | 120 | 161 | 267 | 0.235 | -0.030 |
| stepped_next_gw | 60% | assists | 351 | 464 | 68 | 88 | 142 | 0.126 | -0.043 |
| stepped_next_gw | 80% | goals | 199 | 238 | 67 | 87 | 137 | 0.256 | -0.042 |
| stepped_next_gw | 80% | assists | 192 | 244 | 35 | 46 | 70 | 0.126 | -0.040 |

A positive Δ means stepped next-GW beats the locked full-remainder gate on that cell.
DEF goals leaving 0 under the gate is a primary diagnostic here too.
