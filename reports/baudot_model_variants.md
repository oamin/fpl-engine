# Baudot model variants vs locked gate

Card / FOV / learnable Tversky fixed. Only the learner changes.
Gate method: `baudot_row_mixer_learnable_tversky` · mean score 0.218.

| arm | params | epochs | α | β | mean score | Δ gate | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate (locked) | — | — | 0.447 | 0.553 | 0.218 | +0.000 | — |
| 1_deeper_mixer | 76279 | 27 | 0.450 | 0.550 | 0.156 | -0.062 | 65.8 |
| 2_block_cnn | 44935 | 23 | 0.449 | 0.551 | 0.138 | -0.080 | 42.1 |

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
| 1_deeper_mixer | 20% | goals | 730 | 876 | 216 | 283 | 438 | 0.224 | -0.036 |
| 1_deeper_mixer | 20% | assists | 707 | 879 | 148 | 181 | 297 | 0.150 | -0.010 |
| 1_deeper_mixer | 40% | goals | 544 | 847 | 202 | 267 | 410 | 0.187 | -0.081 |
| 1_deeper_mixer | 40% | assists | 531 | 852 | 140 | 168 | 266 | 0.126 | -0.032 |
| 1_deeper_mixer | 60% | goals | 360 | 648 | 147 | 194 | 308 | 0.162 | -0.104 |
| 1_deeper_mixer | 60% | assists | 351 | 654 | 112 | 133 | 214 | 0.120 | -0.049 |
| 1_deeper_mixer | 80% | goals | 199 | 368 | 86 | 113 | 166 | 0.164 | -0.134 |
| 1_deeper_mixer | 80% | assists | 192 | 373 | 62 | 79 | 115 | 0.113 | -0.053 |
| 2_block_cnn | 20% | goals | 730 | 1320 | 303 | 386 | 612 | 0.163 | -0.097 |
| 2_block_cnn | 20% | assists | 707 | 1322 | 209 | 259 | 436 | 0.110 | -0.049 |
| 2_block_cnn | 40% | goals | 544 | 1003 | 227 | 293 | 463 | 0.159 | -0.109 |
| 2_block_cnn | 40% | assists | 531 | 1005 | 160 | 198 | 322 | 0.110 | -0.048 |
| 2_block_cnn | 60% | goals | 360 | 667 | 150 | 194 | 305 | 0.158 | -0.108 |
| 2_block_cnn | 60% | assists | 351 | 669 | 111 | 132 | 210 | 0.114 | -0.055 |
| 2_block_cnn | 80% | goals | 199 | 342 | 83 | 108 | 160 | 0.179 | -0.120 |
| 2_block_cnn | 80% | assists | 192 | 343 | 54 | 69 | 101 | 0.113 | -0.053 |

A positive Δ means the variant beats the locked gate on that cell.
GKP painted goals by cut are listed per arm in REPORT_STATE.
