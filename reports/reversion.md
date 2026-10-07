Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Mean reversion and an oracle minutes ceiling

No change is made to score_xp.

The serial reversion diagnostic evaluates whether player performance deviations from baseline in preceding weeks predict subsequent forecast errors; the sign convention is strictly actual minus baseline.

A negative regression slope indicates that past outperformance relative to the baseline is followed by an undershoot in the subsequent match.

All conditioning splits (prior minutes, bookmaker team strength, rolling expected minutes) use strictly pre-decision information; same-week minutes never enter reversion conditioning.

The three-week residual is secondary and does not choose a model.

The slope stays above zero for both baselines, on the full sample and when the previous week had at least 60 minutes. That is persistence of the residual, not a bounce through the baseline. No reversion term is added to score_xp.

Doubles are excluded. A missing predecessor is not filled from an older week. Where 2022-23 gameweek 7 is absent, gameweek 8 uses gameweek 6.

## single week, every single-fixture pair

expected points: N=98488. Slope +0.1455 [+0.1282, +0.1610]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 213 | -2.6247 [-3.2743, -1.9812] | the interval stays below zero |
| [-5, -3) | 2237 | -1.7330 [-1.8736, -1.5774] | the interval stays below zero |
| [-3, -2) | 4684 | -0.9905 [-1.0724, -0.9074] | the interval stays below zero |
| [-2, -1) | 10415 | -0.4281 [-0.4785, -0.3774] | the interval stays below zero |
| [-1, 0) | 24303 | +0.0664 [+0.0383, +0.0943] | the interval stays above zero |
| [0, +1) | 46200 | +0.1461 [+0.1343, +0.1582] | the interval stays above zero |
| [+1, +2) | 2868 | +0.6959 [+0.5596, +0.8319] | the interval stays above zero |
| [+2, +3) | 1787 | +0.1532 [+0.0022, +0.3157] | the interval stays above zero |
| [+3, +5) | 2961 | +0.0661 [-0.0836, +0.2258] | the interval covers zero |
| > +5 | 2820 | +0.0812 [-0.1035, +0.2607] | the interval covers zero |

score_xp: N=98488. Slope +0.1072 [+0.0927, +0.1208]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 217 | -1.7749 [-2.3154, -1.2098] | the interval stays below zero |
| [-5, -3) | 2298 | -1.0795 [-1.2305, -0.9095] | the interval stays below zero |
| [-3, -2) | 5245 | -0.6572 [-0.7569, -0.5587] | the interval stays below zero |
| [-2, -1) | 10800 | -0.3027 [-0.3610, -0.2430] | the interval stays below zero |
| [-1, 0) | 24342 | +0.0715 [+0.0426, +0.1005] | the interval stays above zero |
| [0, +1) | 46368 | +0.0939 [+0.0815, +0.1071] | the interval stays above zero |
| [+1, +2) | 1515 | +0.1760 [-0.0488, +0.3861] | the interval covers zero |
| [+2, +3) | 1894 | +0.0437 [-0.1287, +0.2255] | the interval covers zero |
| [+3, +5) | 2932 | +0.0834 [-0.0542, +0.2121] | the interval covers zero |
| > +5 | 2877 | +0.2389 [+0.1028, +0.3835] | the interval stays above zero |

## single week, previous minutes at least 60

expected points: N=27425. Slope +0.0463 [+0.0290, +0.0616]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 105 | undefined | undefined |
| [-5, -3) | 975 | -1.1433 [-1.4201, -0.8912] | the interval stays below zero |
| [-3, -2) | 1936 | -0.4524 [-0.6135, -0.2865] | the interval stays below zero |
| [-2, -1) | 3808 | -0.0016 [-0.1110, +0.1076] | the interval covers zero |
| [-1, 0) | 5289 | +0.4161 [+0.3272, +0.4998] | the interval stays above zero |
| [0, +1) | 5735 | +0.6310 [+0.5419, +0.7180] | the interval stays above zero |
| [+1, +2) | 2733 | +0.7042 [+0.5744, +0.8325] | the interval stays above zero |
| [+2, +3) | 1552 | +0.1511 [-0.0257, +0.3418] | the interval covers zero |
| [+3, +5) | 2646 | +0.0231 [-0.1321, +0.2024] | the interval covers zero |
| > +5 | 2646 | +0.0633 [-0.1221, +0.2493] | the interval covers zero |

score_xp: N=27425. Slope +0.0596 [+0.0444, +0.0747]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 116 | undefined | undefined |
| [-5, -3) | 1249 | -0.5469 [-0.7697, -0.2971] | the interval stays below zero |
| [-3, -2) | 3063 | -0.3645 [-0.5121, -0.2113] | the interval stays below zero |
| [-2, -1) | 5425 | -0.1137 [-0.1943, -0.0247] | the interval stays below zero |
| [-1, 0) | 6207 | -0.0771 [-0.1507, -0.0008] | the interval stays below zero |
| [0, +1) | 3057 | +0.1717 [+0.0592, +0.2767] | the interval stays above zero |
| [+1, +2) | 1376 | +0.1923 [-0.0290, +0.4034] | the interval covers zero |
| [+2, +3) | 1675 | -0.0100 [-0.1990, +0.1698] | the interval covers zero |
| [+3, +5) | 2596 | +0.0326 [-0.1198, +0.1748] | the interval covers zero |
| > +5 | 2661 | +0.1875 [+0.0360, +0.3345] | the interval stays above zero |

## single week, previous minutes at least 60, strong fixture

expected points: N=13043. Slope +0.0519 [+0.0286, +0.0733]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 86 | undefined | undefined |
| [-5, -3) | 599 | -0.7623 [-1.2030, -0.3159] | the interval stays below zero |
| [-3, -2) | 1058 | -0.1559 [-0.4349, +0.1353] | the interval covers zero |
| [-2, -1) | 1875 | +0.3704 [+0.1819, +0.5588] | the interval stays above zero |
| [-1, 0) | 2360 | +0.7368 [+0.5873, +0.8986] | the interval stays above zero |
| [0, +1) | 2444 | +0.8340 [+0.6850, +0.9834] | the interval stays above zero |
| [+1, +2) | 1272 | +0.7663 [+0.5423, +1.0082] | the interval stays above zero |
| [+2, +3) | 752 | +0.3251 [+0.0325, +0.6038] | the interval stays above zero |
| [+3, +5) | 1305 | +0.2265 [-0.0061, +0.4664] | the interval covers zero |
| > +5 | 1292 | +0.3219 [+0.0731, +0.5737] | the interval stays above zero |

score_xp: N=13043. Slope +0.0686 [+0.0455, +0.0901]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 103 | undefined | undefined |
| [-5, -3) | 938 | -0.3121 [-0.6383, +0.0347] | the interval covers zero |
| [-3, -2) | 1969 | -0.2638 [-0.4538, -0.0616] | the interval stays below zero |
| [-2, -1) | 2569 | -0.0675 [-0.2302, +0.0934] | the interval covers zero |
| [-1, 0) | 2155 | +0.0061 [-0.1287, +0.1405] | the interval covers zero |
| [0, +1) | 1238 | +0.3112 [+0.1219, +0.5023] | the interval stays above zero |
| [+1, +2) | 749 | +0.1653 [-0.0957, +0.4553] | the interval covers zero |
| [+2, +3) | 939 | -0.1071 [-0.3611, +0.1432] | the interval covers zero |
| [+3, +5) | 1146 | +0.0923 [-0.1786, +0.3607] | the interval covers zero |
| > +5 | 1237 | +0.2441 [+0.0244, +0.4968] | the interval stays above zero |

## single week, previous minutes at least 60, weak fixture

expected points: N=13057. Slope +0.0416 [+0.0170, +0.0658]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 15 | undefined | undefined |
| [-5, -3) | 345 | -1.5757 [-1.9934, -1.1114] | the interval stays below zero |
| [-3, -2) | 801 | -0.7309 [-0.9843, -0.4702] | the interval stays below zero |
| [-2, -1) | 1758 | -0.4022 [-0.5557, -0.2241] | the interval stays below zero |
| [-1, 0) | 2704 | +0.1315 [+0.0096, +0.2589] | the interval stays above zero |
| [0, +1) | 2974 | +0.4771 [+0.3285, +0.6228] | the interval stays above zero |
| [+1, +2) | 1316 | +0.5909 [+0.4207, +0.7703] | the interval stays above zero |
| [+2, +3) | 707 | -0.0244 [-0.2740, +0.2433] | the interval covers zero |
| [+3, +5) | 1221 | -0.0887 [-0.3054, +0.1172] | the interval covers zero |
| > +5 | 1216 | -0.3518 [-0.6032, -0.0921] | the interval stays below zero |

score_xp: N=13057. Slope +0.0492 [+0.0276, +0.0696]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 12 | undefined | undefined |
| [-5, -3) | 268 | -1.1112 [-1.4205, -0.7507] | the interval stays below zero |
| [-3, -2) | 968 | -0.4500 [-0.6948, -0.2116] | the interval stays below zero |
| [-2, -1) | 2563 | -0.2216 [-0.3619, -0.0768] | the interval stays below zero |
| [-1, 0) | 3740 | -0.1440 [-0.2478, -0.0455] | the interval stays below zero |
| [0, +1) | 1687 | +0.0801 [-0.0491, +0.2191] | the interval covers zero |
| [+1, +2) | 561 | -0.0237 [-0.2720, +0.2151] | the interval covers zero |
| [+2, +3) | 658 | +0.1043 [-0.1148, +0.3673] | the interval covers zero |
| [+3, +5) | 1313 | +0.0418 [-0.1569, +0.2433] | the interval covers zero |
| > +5 | 1287 | +0.0767 [-0.1363, +0.2772] | the interval covers zero |

## single week, previous minutes at least 60, xmi at least 60

expected points: N=22253. Slope +0.0430 [+0.0235, +0.0605]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 102 | undefined | undefined |
| [-5, -3) | 939 | -1.1088 [-1.3880, -0.8493] | the interval stays below zero |
| [-3, -2) | 1826 | -0.4231 [-0.5935, -0.2510] | the interval stays below zero |
| [-2, -1) | 3465 | -0.0114 [-0.1196, +0.1028] | the interval covers zero |
| [-1, 0) | 4482 | +0.4232 [+0.3231, +0.5098] | the interval stays above zero |
| [0, +1) | 4453 | +0.6084 [+0.4996, +0.7170] | the interval stays above zero |
| [+1, +2) | 1698 | +0.5080 [+0.3065, +0.7169] | the interval stays above zero |
| [+2, +3) | 1216 | -0.0073 [-0.2186, +0.1990] | the interval covers zero |
| [+3, +5) | 2070 | -0.0406 [-0.2102, +0.1467] | the interval covers zero |
| > +5 | 2002 | +0.0134 [-0.1799, +0.2081] | the interval covers zero |

score_xp: N=22253. Slope +0.0480 [+0.0310, +0.0651]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 112 | undefined | undefined |
| [-5, -3) | 1202 | -0.5124 [-0.7420, -0.2585] | the interval stays below zero |
| [-3, -2) | 2906 | -0.3485 [-0.4977, -0.1892] | the interval stays below zero |
| [-2, -1) | 4934 | -0.1305 [-0.2162, -0.0371] | the interval stays below zero |
| [-1, 0) | 5007 | -0.1341 [-0.2213, -0.0454] | the interval stays below zero |
| [0, +1) | 1729 | -0.0981 [-0.2698, +0.0713] | the interval covers zero |
| [+1, +2) | 871 | -0.0784 [-0.3762, +0.2328] | the interval covers zero |
| [+2, +3) | 1427 | -0.1311 [-0.3420, +0.0621] | the interval covers zero |
| [+3, +5) | 2034 | -0.0443 [-0.2141, +0.1196] | the interval covers zero |
| > +5 | 2031 | +0.0563 [-0.1059, +0.2107] | the interval covers zero |

## single week, previous minutes at least 60, xmi below 60

expected points: N=5172. Slope +0.0159 [-0.0194, +0.0516]. the interval covers zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 3 | undefined | undefined |
| [-5, -3) | 36 | undefined | undefined |
| [-3, -2) | 110 | undefined | undefined |
| [-2, -1) | 343 | +0.1289 [-0.2680, +0.5433] | the interval covers zero |
| [-1, 0) | 807 | +0.4941 [+0.2860, +0.7048] | the interval stays above zero |
| [0, +1) | 1282 | +0.7884 [+0.6307, +0.9489] | the interval stays above zero |
| [+1, +2) | 1035 | +1.0710 [+0.8761, +1.2593] | the interval stays above zero |
| [+2, +3) | 336 | +0.6065 [+0.2978, +0.9178] | the interval stays above zero |
| [+3, +5) | 576 | +0.3479 [+0.0750, +0.6415] | the interval stays above zero |
| > +5 | 644 | +0.3393 [-0.0037, +0.6818] | the interval covers zero |

score_xp: N=5172. Slope +0.0742 [+0.0431, +0.1051]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 4 | undefined | undefined |
| [-5, -3) | 47 | undefined | undefined |
| [-3, -2) | 157 | -0.5499 [-1.3509, +0.3573] | the interval covers zero |
| [-2, -1) | 491 | -0.0961 [-0.3627, +0.1770] | the interval covers zero |
| [-1, 0) | 1200 | +0.1159 [-0.0428, +0.2715] | the interval covers zero |
| [0, +1) | 1328 | +0.5126 [+0.3696, +0.6712] | the interval stays above zero |
| [+1, +2) | 505 | +0.6136 [+0.3658, +0.8843] | the interval stays above zero |
| [+2, +3) | 248 | +0.5102 [+0.1035, +0.9665] | the interval stays above zero |
| [+3, +5) | 562 | +0.2329 [-0.0036, +0.4864] | the interval covers zero |
| > +5 | 630 | +0.7233 [+0.4382, +1.0536] | the interval stays above zero |

## three preceding weeks

expected points: N=84982. Slope +0.2795 [+0.2570, +0.2995]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 28 | undefined | undefined |
| [-5, -3) | 633 | -2.3547 [-2.6329, -2.0943] | the interval stays below zero |
| [-3, -2) | 2269 | -1.2147 [-1.3483, -1.0735] | the interval stays below zero |
| [-2, -1) | 7073 | -0.5907 [-0.6561, -0.5239] | the interval stays below zero |
| [-1, 0) | 21934 | -0.0665 [-0.0932, -0.0374] | the interval stays below zero |
| [0, +1) | 44071 | +0.1166 [+0.1041, +0.1289] | the interval stays above zero |
| [+1, +2) | 4992 | +0.2531 [+0.1523, +0.3490] | the interval stays above zero |
| [+2, +3) | 2250 | +0.0816 [-0.1292, +0.2646] | the interval covers zero |
| [+3, +5) | 1473 | +0.0984 [-0.1156, +0.3051] | the interval covers zero |
| > +5 | 259 | +0.5940 [+0.0689, +1.1322] | the interval stays above zero |

score_xp: N=84982. Slope +0.1447 [+0.1246, +0.1651]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 19 | undefined | undefined |
| [-5, -3) | 464 | -1.4405 [-1.8078, -1.0890] | the interval stays below zero |
| [-3, -2) | 2186 | -0.6151 [-0.7983, -0.4170] | the interval stays below zero |
| [-2, -1) | 6778 | -0.4181 [-0.5101, -0.3211] | the interval stays below zero |
| [-1, 0) | 24269 | -0.0038 [-0.0307, +0.0240] | the interval covers zero |
| [0, +1) | 43258 | +0.0449 [+0.0299, +0.0588] | the interval stays above zero |
| [+1, +2) | 4111 | -0.0234 [-0.1184, +0.0686] | the interval covers zero |
| [+2, +3) | 2014 | -0.0348 [-0.1899, +0.1388] | the interval covers zero |
| [+3, +5) | 1498 | -0.0040 [-0.1888, +0.1789] | the interval covers zero |
| > +5 | 385 | +0.4597 [-0.0468, +1.0236] | the interval covers zero |

## three preceding weeks, previous minutes at least 60

expected points: N=23699. Slope +0.1152 [+0.0886, +0.1409]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 13 | undefined | undefined |
| [-5, -3) | 158 | -1.9741 [-2.7379, -1.2299] | the interval stays below zero |
| [-3, -2) | 803 | -0.5801 [-0.8598, -0.2791] | the interval stays below zero |
| [-2, -1) | 2548 | -0.0394 [-0.1734, +0.0966] | the interval covers zero |
| [-1, 0) | 5382 | +0.2093 [+0.1240, +0.2939] | the interval stays above zero |
| [0, +1) | 7479 | +0.4439 [+0.3727, +0.5135] | the interval stays above zero |
| [+1, +2) | 3850 | +0.3733 [+0.2475, +0.4874] | the interval stays above zero |
| [+2, +3) | 1896 | +0.1139 [-0.0914, +0.3085] | the interval covers zero |
| [+3, +5) | 1330 | +0.0957 [-0.1265, +0.3167] | the interval covers zero |
| > +5 | 240 | +0.6347 [+0.0853, +1.1774] | the interval stays above zero |

score_xp: N=23699. Slope +0.1018 [+0.0753, +0.1271]. the interval stays above zero.

| previous residual | N | next residual | reading |
|---|---:|---|---|
| < -5 | 14 | undefined | undefined |
| [-5, -3) | 314 | -0.8022 [-1.3381, -0.2873] | the interval stays below zero |
| [-3, -2) | 1543 | -0.3373 [-0.5695, -0.0890] | the interval stays below zero |
| [-2, -1) | 4039 | -0.2237 [-0.3363, -0.1101] | the interval stays below zero |
| [-1, 0) | 6847 | +0.0203 [-0.0563, +0.0905] | the interval covers zero |
| [0, +1) | 4521 | -0.0052 [-0.1142, +0.1012] | the interval covers zero |
| [+1, +2) | 3034 | +0.0377 [-0.0861, +0.1602] | the interval covers zero |
| [+2, +3) | 1686 | -0.0302 [-0.2048, +0.1625] | the interval covers zero |
| [+3, +5) | 1337 | +0.0129 [-0.1797, +0.2056] | the interval covers zero |
| > +5 | 364 | +0.5592 [-0.0103, +1.2464] | the interval covers zero |

## Oracle minutes

Oracle minutes replays use post-match realized minutes to construct a theoretical performance ceiling; they are not pre-deadline forecasts and do not alter the published scoring model.

The published common-state one-week transfers were replayed and matched before these columns were read. The weights are 0.8 and 0.6. They were not searched.

An oracle interval that excludes zero does not demonstrate that `score_xp` beats expected points in pre-deadline decision-making.

Same-fixture minutes beat the historical xmi on this one-week transfer, and the gap versus expected points covers zero.

| score | versus expected points | reading | versus raw score_xp | reading |
|---|---|---|---|---|
| same-fixture minutes | +0.4733 [-0.2443, +1.2445] | the interval covers zero | +0.7634 [+0.3359, +1.2826] | the interval stays above zero |
| weight 0.8 on same-fixture minutes | +0.4351 [-0.2824, +1.1985] | the interval covers zero | +0.7252 [+0.3206, +1.2214] | the interval stays above zero |
| weight 0.6 on same-fixture minutes | +0.3893 [-0.2977, +1.0994] | the interval covers zero | +0.6794 [+0.2977, +1.1605] | the interval stays above zero |

No winner is declared between `score_xp` and `score_exp_points`, and `score_xp` is unchanged.

Gemini kept the residual sign and reviewed the table ([reversion](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).

Bootstrap 1000, seed 0. A bin season under 20 weeks is omitted. Fewer than two seasons leaves the interval undefined.
