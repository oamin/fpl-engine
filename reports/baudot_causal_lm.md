# Baudot causal block-LM vs locked gate

1D club-season stream of schema blocks (HDR + player blocks per GW).
Causal transformer; loss only on **masked remainder event blocks**.
Same Baudot card and learnable Tversky; FOV = full past + masked remainder events.

Model: 4 layers, d=96, 41 tokens/GW.
Gate: `baudot_row_mixer_learnable_tversky` · mean 0.218.

| arm | params | epochs | α | β | mean score | Δ gate | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate (bidirectional mixer) | — | — | 0.447 | 0.553 | 0.218 | +0.000 | — |
| causal_block_lm | 615079 | 15 | 0.447 | 0.553 | 0.118 | -0.101 | 1138.0 |

## Goals at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| causal_block_lm | 0/27/0 | 103/85/6 | 394/906/119 | 233/233/69 |

## Assists at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| causal_block_lm | 4/27/0 | 181/85/6 | 434/910/129 | 88/233/24 |

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
| causal_block_lm | 20% | goals | 730 | 1251 | 194 | 383 | 425 | 0.114 | -0.146 |
| causal_block_lm | 20% | assists | 707 | 1255 | 159 | 310 | 371 | 0.091 | -0.068 |
| causal_block_lm | 40% | goals | 544 | 832 | 129 | 259 | 278 | 0.123 | -0.145 |
| causal_block_lm | 40% | assists | 531 | 835 | 110 | 220 | 246 | 0.102 | -0.056 |
| causal_block_lm | 60% | goals | 360 | 593 | 92 | 176 | 202 | 0.117 | -0.148 |
| causal_block_lm | 60% | assists | 351 | 595 | 74 | 145 | 169 | 0.092 | -0.077 |
| causal_block_lm | 80% | goals | 199 | 239 | 42 | 80 | 82 | 0.160 | -0.139 |
| causal_block_lm | 80% | assists | 192 | 240 | 38 | 67 | 71 | 0.141 | -0.025 |

Primary call: mean vs gate, and whether DEF goals/assists leave 0/0.
