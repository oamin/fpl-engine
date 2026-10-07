Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Decision placebo

Locked before these totals were read. The greedy rule is one same-position free transfer when that week's score gain is positive. It does not bank a second transfer and it does not take a hit. The opening fifteen maximises the same score. A squad that never transfers is a weak baseline: it cannot replace an injury or a blank. The comparison for the score is greedy on `score_xp` against greedy on `score_exp_points`. The shuffle permutes `score_xp` among eligible players inside each gameweek. Seed (0, season index, gameweek). Points are not shuffled. No winner is declared against `ep_next`.

Every greedy transfer was checked against budget, `sell_price`, same position, the club cap, a squad of 15, and eligibility in that week only. Transfers checked: 392. The buy uses that week's score. Eligibility is three prior appearances and expected minutes at least 45, including weeks of 0 minutes. The current week's minutes and points do not enter it.

A positive number is points per gameweek.

| contrast | pooled mean [95% interval] | reading |
|---|---:|---|
| greedy score_xp − shuffled greedy | +15.0519 [+11.9776, +17.9196] | the interval stays above zero |
| greedy score_xp − greedy expected points | -1.6667 [-4.4446, +1.0552] | the interval covers zero |
| shuffled greedy − shuffled hold | +1.8444 [-0.4613, +4.1631] | the interval covers zero |
| score_xp greedy − score_xp hold | +7.3556 [+4.4961, +10.0446] | the interval stays above zero |

The placebo contrast pairs the realised weekly points of the `score_xp` greedy squad against the shuffled-score greedy squad on the same gameweeks. It is not a difference of the two hold baselines. Each squad builds its own opening fifteen on the column it sees. The two opening squads differ by +9.54 points a week, and the two greedy-minus-hold edges differ by +5.51. That second figure has no interval of its own. The direct gap is those two pieces added, and its interval is the one in the table. It is the real score against a squad built on noise. It is not a win over expected points. Subtracting the published hold rows is not the contrast. The hold row is the squad that never transfers. It is not the skill comparison. Permuting the score inside each gameweek does not reproduce a pooled gain that stays above zero. 2023-24 of that shuffle is several points, so replacing players can beat a squad that never transfers when the score is noise. The skill comparison covers zero. 2022-23 is sharply negative and stays in the pool. The one-week slope's interval reaches 1, so a point estimate below 1 is a direction, not a finding that the predictions were overstated. No winner is declared.

## By season

### greedy score_xp − greedy expected points

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 33 | -14.45 |
| 2023-24 | 34 | +3.03 |
| 2024-25 | 34 | +1.09 |
| 2025-26 | 34 | +3.29 |

### shuffled greedy − shuffled hold

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 33 | +0.15 |
| 2023-24 | 34 | +7.85 |
| 2024-25 | 34 | -2.79 |
| 2025-26 | 34 | +2.12 |

### greedy score_xp − shuffled greedy

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 33 | +6.00 |
| 2023-24 | 34 | +18.47 |
| 2024-25 | 34 | +20.50 |
| 2025-26 | 34 | +14.97 |

### score_xp greedy − score_xp hold

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 33 | +6.97 |
| 2023-24 | 34 | +13.68 |
| 2024-25 | 34 | +4.68 |
| 2025-26 | 34 | +4.09 |

## 2022-23, week by week

This is greedy on `score_xp` minus greedy on expected points. Gameweek 7 has no sheet, so it is absent. Gameweek 16 kicks off on 12 November 2022 and gameweek 17 on 26 December 2022.

| gameweek | points |
|---:|---:|
| 5 | +1.0 |
| 6 | -19.0 |
| 8 | -16.0 |
| 9 | -45.0 |
| 10 | -4.0 |
| 11 | +2.0 |
| 12 | +6.0 |
| 13 | -1.0 |
| 14 | -3.0 |
| 15 | -37.0 |
| 16 | -44.0 |
| 17 | -25.0 |
| 18 | -19.0 |
| 19 | -38.0 |
| 20 | -10.0 |
| 21 | -33.0 |
| 22 | +16.0 |
| 23 | -21.0 |
| 24 | -8.0 |
| 25 | -48.0 |
| 26 | +3.0 |
| 27 | -44.0 |
| 28 | +7.0 |
| 29 | -69.0 |
| 30 | -21.0 |
| 31 | -3.0 |
| 32 | -3.0 |
| 33 | -7.0 |
| 34 | +6.0 |
| 35 | +14.0 |
| 36 | -13.0 |
| 37 | -2.0 |
| 38 | +1.0 |

The season mean is -14.45 on 33 weeks. The loss is in the block before the World Cup break and in the block after it. It is not one missing gameweek and it is not dropped from the pool.

Sheet rows with no opening-odds fixture: 2022-23 0 of 26505; 2023-24 0 of 29725; 2024-25 0 of 27283; 2025-26 0 of 29757. A miss of zero means the World Cup break is a gap in the calendar, not a dropped join.

## Template

The price-target template is a fixed heuristic, not an ownership squad. `selected` is not an input. It is built once, at the first week a legal squad exists, by walking the locked price targets and taking the closest eligible price in that position. Ties break toward the higher score, then the lower player id. Later weeks do not transfer. The official average entry score for 2025-26, gameweeks 5–38, is 49.38. On those weeks this replay's hold averages 50.62, the template 39.06, and greedy 54.71. The official average includes automatic substitutes, chips, and hits. This replay is the eleven plus the captain, with no hit. 2022-23, 2023-24, and 2024-25 are not in that public file. Anchor for those seasons: null.

## What 4.33 and 0.40 were

Three-week realised gain on the one-week predicted gain: a = -1.1392, b = +1.1880 [+0.5917, +1.9721], n = 131.

The old printed threshold used this three-week line: (4 − a) / b = (4 − (-1.1392)) / +1.1880 = +4.326. That is not the hurdle. A slope near 1.2 is what a three-week sum against a one-week prediction looks like. b/3 = +0.3960 is a per-week arithmetic rate rescaling of that slope, not winner's-curse shrinkage.

The same transfers, realised in the decision week only: a = +1.1641, b = +0.4892 [+0.0965, +1.0222], n = 131.

The one-week intercept is +1.1641. The in-sample hurdle solves a1 + b1 x = 4, so x = (4 − a1) / b1 = (4 − (+1.1641)) / +0.4892 = +5.797. That pooled line is an in-sample observation. It was not used to hurdle or execute a transfer.

A threshold that is offered as a choice is fit on the other three seasons. It is undefined when that slope is not positive. None of these numbers entered the rule.

| held-out season | training transfers | a1 | b1 | threshold |
|---|---:|---:|---:|---:|
| 2022-23 | 99 | -0.054 | +0.810 | +5.00 |
| 2023-24 | 98 | +1.394 | +0.517 | +5.04 |
| 2024-25 | 98 | +1.420 | +0.357 | +7.23 |
| 2025-26 | 98 | +1.190 | +0.458 | +6.14 |

No winner is declared between `score_xp` and `score_exp_points`. At 20 live weeks, an interval that covers zero is undetermined. The test continues through gameweek 38.

## Spearman, every week

Two different correlations. The first is `score_xp` against scraped xP. A positive value means the two forecasts rank players the same way. An undefined week has no variation in one column, which is what an all-zero scrape looks like. Undefined weeks: 36. Negative weeks for that correlation: 2025-26 GW1 (-0.055).

The withdrawn −0.35 is not that correlation. It is, on eligible players, Spearman(`score_xp`, points) minus Spearman(scraped xP, points), within position, averaged across positions. On 106 defined weeks the mean within-position Spearman with points is +0.234 for `score_xp` and +0.590 for scraped xP. On filled weeks from gameweek 5 to 38 those levels are +0.233 and +0.587. On filled weeks from gameweek 5 to 38 the mean of those weekly gaps is -0.3539. Negative rank-gap weeks: 2022-23 GW4 (-0.385), 2022-23 GW5 (-0.254), 2022-23 GW6 (-0.479), 2022-23 GW8 (-0.476), 2022-23 GW9 (-0.649), 2022-23 GW10 (-0.518), 2022-23 GW11 (-0.428), 2022-23 GW13 (-0.269), 2022-23 GW14 (-0.075), 2022-23 GW15 (-0.519), 2022-23 GW16 (-0.437), 2022-23 GW17 (-0.761), 2022-23 GW18 (-0.621), 2022-23 GW19 (-0.379), 2022-23 GW20 (-0.286), 2022-23 GW21 (-0.292), 2022-23 GW22 (-0.497), 2022-23 GW23 (-0.344), 2022-23 GW24 (-0.702), 2022-23 GW25 (-0.117), 2022-23 GW26 (-0.268), 2022-23 GW27 (-0.207), 2022-23 GW28 (-0.112), 2022-23 GW29 (-0.282), 2022-23 GW30 (-0.410), 2022-23 GW31 (-0.534), 2022-23 GW32 (-0.148), 2022-23 GW33 (-0.355), 2022-23 GW34 (-0.069), 2022-23 GW35 (-0.299), 2022-23 GW37 (-0.325), 2022-23 GW38 (-0.269), 2023-24 GW4 (-0.274), 2023-24 GW5 (-0.413), 2023-24 GW6 (-0.303), 2023-24 GW7 (-0.382), 2023-24 GW8 (-0.303), 2023-24 GW9 (-0.104), 2023-24 GW10 (-0.333), 2023-24 GW11 (-0.490), 2023-24 GW12 (-0.269), 2023-24 GW13 (-0.699), 2023-24 GW14 (-0.258), 2023-24 GW15 (-0.466), 2023-24 GW16 (-0.450), 2023-24 GW17 (-0.287), 2023-24 GW18 (-0.384), 2023-24 GW19 (-0.312), 2023-24 GW20 (-0.228), 2023-24 GW21 (-0.460), 2023-24 GW22 (-0.459), 2023-24 GW23 (-0.343), 2023-24 GW24 (-0.386), 2023-24 GW25 (-0.232), 2023-24 GW27 (-0.264), 2023-24 GW28 (-0.300), 2023-24 GW29 (-0.733), 2023-24 GW30 (-0.457), 2023-24 GW31 (-0.282), 2023-24 GW32 (-0.107), 2023-24 GW33 (-0.455), 2023-24 GW34 (-0.195), 2023-24 GW35 (-0.203), 2023-24 GW36 (-0.188), 2023-24 GW37 (-0.140), 2023-24 GW38 (-0.152), 2024-25 GW4 (-0.534), 2024-25 GW5 (-0.356), 2024-25 GW6 (-0.345), 2024-25 GW7 (-0.383), 2024-25 GW8 (-0.490), 2024-25 GW9 (-0.363), 2024-25 GW10 (-0.498), 2024-25 GW11 (-0.333), 2024-25 GW12 (-0.518), 2024-25 GW13 (-0.142), 2024-25 GW14 (-0.266), 2024-25 GW15 (-0.335), 2024-25 GW16 (-0.479), 2024-25 GW17 (-0.410), 2024-25 GW18 (-0.382), 2024-25 GW19 (-0.241), 2024-25 GW20 (-0.354), 2024-25 GW21 (-0.118), 2024-25 GW23 (-0.283), 2024-25 GW24 (-0.444), 2024-25 GW25 (-0.379), 2024-25 GW26 (-0.544), 2024-25 GW27 (-0.134), 2024-25 GW28 (-0.400), 2024-25 GW29 (-0.279), 2024-25 GW30 (-0.479), 2024-25 GW31 (-0.434), 2024-25 GW33 (-0.190), 2024-25 GW35 (-0.201), 2024-25 GW36 (-0.473), 2024-25 GW37 (-0.215), 2024-25 GW38 (-0.242), 2025-26 GW4 (-0.368), 2025-26 GW5 (-0.503), 2025-26 GW6 (-0.524), 2025-26 GW8 (-0.469), 2025-26 GW9 (-0.555), 2025-26 GW24 (-0.284), 2025-26 GW29 (-0.293), 2025-26 GW38 (-0.344). 106 of 106 defined weeks. On filled scrapes, 94.4% of players who played have a nonzero scraped xP and 23.7% of players who did not. A pre-match forecast does not know who played. Scraped xP stays unusable as a feature and as a benchmark.

2022-23 has no gameweek 7 sheet, so that week is absent rather than negative. An all-zero scraped column is undefined, not a negative correlation. The rank gap is negative on every defined week, so it is not a few misaligned weeks and not the missing gameweek 7. These weeks are a diagnostic of a withdrawn figure. They are not a benchmark. Gemini reviewed the diagnostics ([placebo diagnostics](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).
