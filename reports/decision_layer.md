Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Decision layer

First batch, locked before these totals were read. The score column is `score_xp`. `ep_next` is the live column captured beside it. Live pre-deadline gameweeks so far: 0. No winner is declared between the two. Chips did not fire. The captain baseline is the highest `score_exp_points` inside the hold eleven, not the highest realised points.

## Alignment of the two forecasts

One closed gameweek, 2024-25 gameweek 10. This is a diagnostic of the join, not a benchmark. The withdrawn −0.35 is Spearman(score_xp, points) minus Spearman(scraped xP, points). It is not the correlation of the two forecasts with each other.

Rows 674, players 674, paired 674. Duplicate player-fixture keys: 0. Max rows for one player: 1. Same-row join: True. Positions legal: True. Gameweek matches: True.

Spearman(score_xp, scraped xP) +0.7236. Pearson +0.7452. Spearman(−score_xp, scraped xP) -0.7236. A sign flip fits better: False.

Spearman(score_xp, points) +0.7360. Spearman(scraped xP, points) +0.7700. Gap -0.0340.

Scatter: `data/plots/alignment_2024_25_gw10.png`.

## Power of 20 gameweeks

Residuals are the certified score_xp minus expected-points gameweek deltas with the mean removed. Each draw takes 20 residuals, adds a candidate effect, and reruns the gameweek cluster interval (B = 1000, seed 0, 400 draws). Power is the share of intervals that lie entirely above zero.

Smallest effect with power at least 80% at 20 gameweeks: 0.0100. Median half-width of the null intervals: +0.0055. Residual sd of the closed-season deltas: +0.0143 (135 gameweeks).

The full-sample interval on about 130 gameweeks is not this resolution. A gap as small as the withdrawn −0.0016 log score is below what 20 weeks resolve.

## Baselines

Shared start: the budgeted fifteen that maximises score_xp, and a template built the same week from the locked price targets. Hold keeps that fifteen. Greedy may make one same-position free transfer when the score gain is positive. It does not bank a second transfer and it does not take a hit. That rule is not the published hold margin of 1.25. The template never transfers. A week that cannot fill both fifteens is skipped. Gameweeks in the table: 2022-23 33, 2023-24 34, 2024-25 34, 2025-26 34.

A positive mean is points per gameweek for the first name.

| season | weeks | hold | greedy | template |
|---|---:|---:|---:|---:|
| 2022-23 | 33 | 38.61 | 45.58 | 41.82 |
| 2023-24 | 34 | 40.18 | 53.85 | 37.00 |
| 2024-25 | 34 | 50.82 | 55.50 | 31.68 |
| 2025-26 | 34 | 50.62 | 54.71 | 39.06 |

| comparison | mean [95% interval] |
|---|---:|
| greedy − hold | +7.3556 [+4.4961, +10.0446] |
| greedy − template | +15.1037 [+11.5241, +18.4967] |
| hold − template | +7.7481 [+4.6815, +10.9785] |
| captain score_xp − highest score_exp_points | -0.2074 [-0.7704, +0.2965] |

## Transfer calibration

Realised gain is points of the player in, minus points of the player out, over the decision week and the next two gameweeks that exist. Ordinary least squares of that sum on the predicted score gain: a = -1.1392, b = +1.1880, interval for b [+0.5917, +1.9721], n = 131.

The implied one-week hit threshold is +4.326 points and the per-week shrinkage is +0.3960. Neither number was used to choose a transfer.

## Chips

The chip map is empty. Chips did not fire. Free Hit 12 and Wildcard 16 were not searched.

## Review

The formulas were locked before the run (bc-ffc0ced9). The diagnostics were reviewed after it (bc-3d00af39). The weekly gap is not a captain counted twice and not a double gameweek summed twice. The Gameweek 10 correlation shows the two forecasts move together. It does not explain the pooled −0.35 rank gap against points. 0.010 is the effect 20 gameweeks detect at 80% power. The null half-width is about half of that, which is a weaker bar. The hit threshold is an observation. It was not used to choose a transfer.
