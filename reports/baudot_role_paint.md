# Baudot role paint vs locked gate (DEF goals / assists)

Card / FOV fixed. Shared paint head → role-specific paint heads (role from card bits).
Equal per-position Tversky stays parked (forced DEF volume, no hits, GKP noise).
Gate method: `baudot_row_mixer_learnable_tversky` · mean score 0.218.

| arm | params | epochs | mean score | Δ gate | seconds |
|---|---:|---:|---:|---:|---:|
| gate (locked) | — | — | 0.218 | +0.000 | — |
| 1_role_paint | 39145 | 20 | 0.170 | -0.048 | 40.8 |
| 2_role_paint_dual | 39146 | 20 | 0.181 | -0.037 | 41.1 |
| 3_def_goal_aux | 39145 | 27 | 0.157 | -0.061 | 54.5 |

## Goals at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| gate_replay | 0/0/0 | 103/0/0 | 394/386/83 | 233/441/133 |
| 1_role_paint | 0/0/0 | 103/0/0 | 394/483/87 | 233/364/123 |
| 2_role_paint_dual | 0/0/0 | 103/0/0 | 394/466/87 | 233/368/125 |
| 3_def_goal_aux | 0/0/0 | 103/78/5 | 394/641/119 | 233/441/142 |

## Assists at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| gate_replay | 4/0/0 | 181/0/0 | 434/404/77 | 88/445/58 |
| 1_role_paint | 4/0/0 | 181/0/0 | 434/567/101 | 88/304/43 |
| 2_role_paint_dual | 4/0/0 | 181/0/0 | 434/658/117 | 88/258/40 |
| 3_def_goal_aux | 4/0/0 | 181/0/0 | 434/843/157 | 88/272/37 |

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
| 1_role_paint | 20% | goals | 730 | 847 | 210 | 296 | 434 | 0.230 | -0.031 |
| 1_role_paint | 20% | assists | 707 | 871 | 144 | 190 | 309 | 0.148 | -0.011 |
| 1_role_paint | 40% | goals | 544 | 765 | 181 | 249 | 378 | 0.197 | -0.071 |
| 1_role_paint | 40% | assists | 531 | 778 | 135 | 171 | 272 | 0.141 | -0.018 |
| 1_role_paint | 60% | goals | 360 | 547 | 124 | 170 | 260 | 0.180 | -0.086 |
| 1_role_paint | 60% | assists | 351 | 568 | 106 | 130 | 204 | 0.143 | -0.026 |
| 1_role_paint | 80% | goals | 199 | 320 | 77 | 102 | 146 | 0.185 | -0.114 |
| 1_role_paint | 80% | assists | 192 | 324 | 59 | 78 | 113 | 0.136 | -0.031 |
| 2_role_paint_dual | 20% | goals | 730 | 834 | 212 | 296 | 422 | 0.237 | -0.023 |
| 2_role_paint_dual | 20% | assists | 707 | 916 | 157 | 213 | 328 | 0.149 | -0.010 |
| 2_role_paint_dual | 40% | goals | 544 | 722 | 175 | 250 | 359 | 0.208 | -0.060 |
| 2_role_paint_dual | 40% | assists | 531 | 759 | 133 | 174 | 261 | 0.144 | -0.014 |
| 2_role_paint_dual | 60% | goals | 360 | 519 | 121 | 173 | 249 | 0.191 | -0.075 |
| 2_role_paint_dual | 60% | assists | 351 | 532 | 102 | 130 | 190 | 0.152 | -0.016 |
| 2_role_paint_dual | 80% | goals | 199 | 283 | 73 | 102 | 134 | 0.213 | -0.085 |
| 2_role_paint_dual | 80% | assists | 192 | 291 | 55 | 74 | 100 | 0.150 | -0.016 |
| 3_def_goal_aux | 20% | goals | 730 | 1160 | 266 | 372 | 527 | 0.177 | -0.083 |
| 3_def_goal_aux | 20% | assists | 707 | 1115 | 194 | 267 | 386 | 0.135 | -0.025 |
| 3_def_goal_aux | 40% | goals | 544 | 902 | 202 | 286 | 408 | 0.169 | -0.099 |
| 3_def_goal_aux | 40% | assists | 531 | 871 | 149 | 207 | 290 | 0.130 | -0.029 |
| 3_def_goal_aux | 60% | goals | 360 | 617 | 140 | 192 | 279 | 0.167 | -0.098 |
| 3_def_goal_aux | 60% | assists | 351 | 583 | 108 | 142 | 197 | 0.139 | -0.030 |
| 3_def_goal_aux | 80% | goals | 199 | 322 | 81 | 107 | 146 | 0.192 | -0.106 |
| 3_def_goal_aux | 80% | assists | 192 | 308 | 59 | 78 | 100 | 0.147 | -0.019 |

Primary call: does DEF goals leave 0/0, and do assist hits rise without GKP goal noise or a mean-score collapse vs the locked gate.
