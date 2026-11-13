# Baudot mixer ablations (one-by-one)

Full past + one-shot remainder FOV. Metrics: `tally_row` + `balanced_score`.

| arm | loss | train | per-pos | epochs | best mean score | seconds |
|---|---|---|---|---:|---:|---:|
| 1_soft_dice_20 | soft_dice | 20ep constant | False | 20 | 0.185 | 38.3 |
| 2_tversky_20 | tversky | 20ep constant | False | 20 | 0.201 | 38.1 |
| 3_tversky_long | tversky | 50ep cosine+early | False | 20 | 0.204 | 38.2 |
| 4_tversky_long_pos | tversky | 50ep cosine+early | True | 29 | 0.171 | 56.4 |

## Holdout table

| arm | revealed | channel | actual events | painted events | hits | actual ±1 | painted ±1 | score |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1_soft_dice_20 | 20% | goals | 730 | 914 | 240 | 322 | 472 | 0.233 |
| 1_soft_dice_20 | 20% | assists | 707 | 904 | 150 | 190 | 301 | 0.146 |
| 1_soft_dice_20 | 40% | goals | 544 | 729 | 183 | 252 | 374 | 0.215 |
| 1_soft_dice_20 | 40% | assists | 531 | 721 | 117 | 151 | 228 | 0.138 |
| 1_soft_dice_20 | 60% | goals | 360 | 498 | 124 | 172 | 255 | 0.209 |
| 1_soft_dice_20 | 60% | assists | 351 | 491 | 85 | 104 | 158 | 0.144 |
| 1_soft_dice_20 | 80% | goals | 199 | 257 | 69 | 97 | 131 | 0.234 |
| 1_soft_dice_20 | 80% | assists | 192 | 251 | 47 | 61 | 79 | 0.162 |
| 2_tversky_20 | 20% | goals | 730 | 731 | 196 | 275 | 391 | 0.268 |
| 2_tversky_20 | 20% | assists | 707 | 732 | 112 | 158 | 234 | 0.150 |
| 2_tversky_20 | 40% | goals | 544 | 611 | 154 | 220 | 322 | 0.237 |
| 2_tversky_20 | 40% | assists | 531 | 612 | 94 | 128 | 192 | 0.143 |
| 2_tversky_20 | 60% | goals | 360 | 412 | 101 | 149 | 220 | 0.229 |
| 2_tversky_20 | 60% | assists | 351 | 412 | 74 | 96 | 138 | 0.165 |
| 2_tversky_20 | 80% | goals | 199 | 210 | 58 | 80 | 110 | 0.269 |
| 2_tversky_20 | 80% | assists | 192 | 212 | 32 | 50 | 65 | 0.143 |
| 3_tversky_long | 20% | goals | 730 | 740 | 198 | 275 | 383 | 0.266 |
| 3_tversky_long | 20% | assists | 707 | 740 | 117 | 173 | 247 | 0.155 |
| 3_tversky_long | 40% | goals | 544 | 609 | 152 | 222 | 316 | 0.236 |
| 3_tversky_long | 40% | assists | 531 | 608 | 95 | 136 | 193 | 0.146 |
| 3_tversky_long | 60% | goals | 360 | 424 | 104 | 150 | 216 | 0.225 |
| 3_tversky_long | 60% | assists | 351 | 422 | 75 | 100 | 142 | 0.161 |
| 3_tversky_long | 80% | goals | 199 | 208 | 58 | 81 | 104 | 0.273 |
| 3_tversky_long | 80% | assists | 192 | 208 | 37 | 51 | 68 | 0.171 |
| 4_tversky_long_pos | 20% | goals | 730 | 877 | 174 | 275 | 344 | 0.180 |
| 4_tversky_long_pos | 20% | assists | 707 | 703 | 105 | 183 | 212 | 0.148 |
| 4_tversky_long_pos | 40% | goals | 544 | 670 | 131 | 210 | 265 | 0.175 |
| 4_tversky_long_pos | 40% | assists | 531 | 528 | 78 | 137 | 159 | 0.146 |
| 4_tversky_long_pos | 60% | goals | 360 | 453 | 93 | 143 | 185 | 0.182 |
| 4_tversky_long_pos | 60% | assists | 351 | 367 | 61 | 99 | 115 | 0.163 |
| 4_tversky_long_pos | 80% | goals | 199 | 210 | 49 | 76 | 87 | 0.227 |
| 4_tversky_long_pos | 80% | assists | 192 | 164 | 30 | 54 | 55 | 0.144 |

## Goals at 20% cut by position (actual / painted / hits)

| arm | GKP a/p/h | DEF a/p/h | MID a/p/h | FWD a/p/h |
|---|---|---|---|---|
| 1_soft_dice_20 | 0/0/0 | 103/0/0 | 394/523/112 | 233/391/128 |
| 2_tversky_20 | 0/0/0 | 103/0/0 | 394/329/67 | 233/402/129 |
| 3_tversky_long | 0/0/0 | 103/0/0 | 394/379/78 | 233/361/120 |
| 4_tversky_long_pos | 0/70/0 | 103/127/4 | 394/457/88 | 233/223/82 |

Arms: (1) soft_dice baseline → (2) Tversky β=0.7 α=0.3 → (3) longer cosine+early-stop → (4) + equal per-position Tversky.
