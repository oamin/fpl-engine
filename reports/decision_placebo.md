Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Decision placebo

Locked before these totals were read. The greedy rule is one same-position free transfer when that week's score gain is positive. It does not bank a second transfer and it does not take a hit. The opening fifteen maximises the same score. A squad that never transfers is a weak baseline: it cannot replace an injury or a blank. The comparison for the score is greedy on `score_xp` against greedy on `score_exp_points`. The shuffle permutes `score_xp` among eligible players inside each gameweek. Seed (0, season index, gameweek). Points are not shuffled. No winner is declared against `ep_next`.

Every greedy transfer was checked against budget, `sell_price`, same position, the club cap, a squad of 15, and eligibility in that week only. Transfers checked: 392. The buy uses that week's score. Eligibility is three prior appearances and expected minutes at least 45, including weeks of 0 minutes. The current week's minutes and points do not enter it.

A positive number is points per gameweek.

| contrast | pooled mean [95% interval] | reading |
|---|---:|---|
| greedy score_xp − greedy expected points | -1.6667 [-4.4446, +1.0552] | the interval covers zero |
| shuffled greedy − shuffled hold | +1.8444 [-0.4613, +4.1631] | the interval covers zero |
| score_xp greedy − score_xp hold | +7.3556 [+4.4961, +10.0446] | the interval stays above zero |

The hold row is the squad that never transfers. It is not the skill comparison. Permuting the score inside each gameweek does not reproduce a pooled gain that stays above zero. 2023-24 of that shuffle is several points, so replacing players can beat a squad that never transfers when the score is noise. The skill comparison covers zero. 2022-23 is sharply negative and stays in the pool. The one-week slope's interval reaches 1, so a point estimate below 1 is a direction, not a finding that the predictions were overstated. No winner is declared.

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

### score_xp greedy − score_xp hold

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 33 | +6.97 |
| 2023-24 | 34 | +13.68 |
| 2024-25 | 34 | +4.68 |
| 2025-26 | 34 | +4.09 |

## What 4.33 and 0.40 were

Three-week realised gain on the one-week predicted gain: a = -1.1392, b = +1.1880 [+0.5917, +1.9721], n = 131.

The printed threshold solves a + b x = 4, so x = (4 − a) / b = (4 − (-1.1392)) / +1.1880 = +4.326. Dividing 4 by b would be that number only if the intercept were zero. b/3 = +0.3960 rescales the three-week slope to a per-week rate. It is not an estimate that high predictions should be shrunk for winner's curse.

The same transfers, realised in the decision week only: a = +1.1641, b = +0.4892 [+0.0965, +1.0222], n = 131.

The one-week slope is the one that would sit below 1 if selecting the largest predicted gain overstated the points. It is not fed back into the rule. At 20 live weeks, an interval that covers zero is undetermined. The test continues through gameweek 38.

## Spearman, every week

Two different correlations. The first is `score_xp` against scraped xP. A positive value means the two forecasts rank players the same way. An undefined week has no variation in one column, which is what an all-zero scrape looks like. Undefined weeks: 36. Negative weeks for that correlation: 2025-26 GW1 (-0.055).

The withdrawn −0.35 is not that correlation. It is, on eligible players, Spearman(`score_xp`, points) minus Spearman(scraped xP, points), within position, averaged across positions. On filled weeks from gameweek 5 to 38 the mean of those weekly gaps is -0.3539. Negative rank-gap weeks: 2022-23 GW4 (-0.385), 2022-23 GW5 (-0.254), 2022-23 GW6 (-0.479), 2022-23 GW8 (-0.476), 2022-23 GW9 (-0.649), 2022-23 GW10 (-0.518), 2022-23 GW11 (-0.428), 2022-23 GW13 (-0.269), 2022-23 GW14 (-0.075), 2022-23 GW15 (-0.519), 2022-23 GW16 (-0.437), 2022-23 GW17 (-0.761), 2022-23 GW18 (-0.621), 2022-23 GW19 (-0.379), 2022-23 GW20 (-0.286), 2022-23 GW21 (-0.292), 2022-23 GW22 (-0.497), 2022-23 GW23 (-0.344), 2022-23 GW24 (-0.702), 2022-23 GW25 (-0.117), 2022-23 GW26 (-0.268), 2022-23 GW27 (-0.207), 2022-23 GW28 (-0.112), 2022-23 GW29 (-0.282), 2022-23 GW30 (-0.410), 2022-23 GW31 (-0.534), 2022-23 GW32 (-0.148), 2022-23 GW33 (-0.355), 2022-23 GW34 (-0.069), 2022-23 GW35 (-0.299), 2022-23 GW37 (-0.325), 2022-23 GW38 (-0.269), 2023-24 GW4 (-0.274), 2023-24 GW5 (-0.413), 2023-24 GW6 (-0.303), 2023-24 GW7 (-0.382), 2023-24 GW8 (-0.303), 2023-24 GW9 (-0.104), 2023-24 GW10 (-0.333), 2023-24 GW11 (-0.490), 2023-24 GW12 (-0.269), 2023-24 GW13 (-0.699), 2023-24 GW14 (-0.258), 2023-24 GW15 (-0.466), 2023-24 GW16 (-0.450), 2023-24 GW17 (-0.287), 2023-24 GW18 (-0.384), 2023-24 GW19 (-0.312), 2023-24 GW20 (-0.228), 2023-24 GW21 (-0.460), 2023-24 GW22 (-0.459), 2023-24 GW23 (-0.343), 2023-24 GW24 (-0.386), 2023-24 GW25 (-0.232), 2023-24 GW27 (-0.264), 2023-24 GW28 (-0.300), 2023-24 GW29 (-0.733), 2023-24 GW30 (-0.457), 2023-24 GW31 (-0.282), 2023-24 GW32 (-0.107), 2023-24 GW33 (-0.455), 2023-24 GW34 (-0.195), 2023-24 GW35 (-0.203), 2023-24 GW36 (-0.188), 2023-24 GW37 (-0.140), 2023-24 GW38 (-0.152), 2024-25 GW4 (-0.534), 2024-25 GW5 (-0.356), 2024-25 GW6 (-0.345), 2024-25 GW7 (-0.383), 2024-25 GW8 (-0.490), 2024-25 GW9 (-0.363), 2024-25 GW10 (-0.498), 2024-25 GW11 (-0.333), 2024-25 GW12 (-0.518), 2024-25 GW13 (-0.142), 2024-25 GW14 (-0.266), 2024-25 GW15 (-0.335), 2024-25 GW16 (-0.479), 2024-25 GW17 (-0.410), 2024-25 GW18 (-0.382), 2024-25 GW19 (-0.241), 2024-25 GW20 (-0.354), 2024-25 GW21 (-0.118), 2024-25 GW23 (-0.283), 2024-25 GW24 (-0.444), 2024-25 GW25 (-0.379), 2024-25 GW26 (-0.544), 2024-25 GW27 (-0.134), 2024-25 GW28 (-0.400), 2024-25 GW29 (-0.279), 2024-25 GW30 (-0.479), 2024-25 GW31 (-0.434), 2024-25 GW33 (-0.190), 2024-25 GW35 (-0.201), 2024-25 GW36 (-0.473), 2024-25 GW37 (-0.215), 2024-25 GW38 (-0.242), 2025-26 GW4 (-0.368), 2025-26 GW5 (-0.503), 2025-26 GW6 (-0.524), 2025-26 GW8 (-0.469), 2025-26 GW9 (-0.555), 2025-26 GW24 (-0.284), 2025-26 GW29 (-0.293), 2025-26 GW38 (-0.344). 106 of 106 defined weeks.

2022-23 has no gameweek 7 sheet, so that week is absent rather than negative. An all-zero scraped column is undefined, not a negative correlation. The rank gap is negative on every defined week, so it is not a few misaligned weeks and not the missing gameweek 7. These weeks are a diagnostic of a withdrawn figure. They are not a benchmark. Gemini reviewed the diagnostics ([placebo diagnostics](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).
