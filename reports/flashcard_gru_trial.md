# Flash-card GRU trial

Exploratory. One held-out season. Not a gated score comparison.
2026-27 was not read.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Cards with at least 12 fixtures: train 2348, test 818.
Test cut: first 50% of fixture columns visible.
Minutes, home, and opponent stay visible on the remainder.
Event bars on the remainder are hidden. Heights are 0, 1, 2, or 3+ (clean sheet 0 or 1).

## Fit

One-layer GRU, hidden 64. The classifier also sees minutes, a benched flag,
a played flag (60 minutes or more), and home.
Teacher forcing. Cross-entropy on the hidden columns.
Prefix fraction during training is uniform between 30% and 70%.
Epoch mean NLL per hidden bar: 0.481, 0.263, 0.242, 0.234, 0.230, 0.229, 0.222, 0.223, 0.219, 0.215.

Baseline: the player's prefix histogram, shrunk toward a position × minutes-bin
prior fit on the training seasons only.

## Held-out negative log-likelihood

Lower is better. Played columns are remainder fixtures with at least 60 minutes.
Blank means the remainder event bars are not fed back in. Teacher forcing feeds
true earlier remainder bars, so that number is an easier next-step score.

| model | goals | assists | clean sheets | goals conceded |
|---|---:|---:|---:|---:|
| GRU blank, all columns | 0.654 | 0.569 | 0.653 | 0.626 |
| GRU blank, played | 1.333 | 1.261 | 1.813 | 1.659 |
| GRU teacher, played | 0.320 | 0.329 | 0.607 | 1.357 |
| Baseline, all columns | 0.102 | 0.104 | 0.151 | 0.448 |
| Baseline, played | 0.321 | 0.329 | 0.585 | 1.330 |

## What a played week looks like

Rate is the share of played remainder columns whose true bar is at least 1.
P is the blank-remainder GRU's mean probability of that event.
Argmax sum is the total height of the single most likely chart.

| channel | actual rate | GRU P | mean actual sum | mean sample sum | mean argmax sum | baseline expected sum |
|---|---:|---:|---:|---:|---:|---:|
| goals | 0.099 | 0.663 | 0.60 | 0.64 | 0.00 | 0.66 |
| assists | 0.097 | 0.669 | 0.56 | 0.67 | 0.00 | 0.60 |
| clean_sheets | 0.272 | 0.860 | 1.27 | 1.25 | 0.07 | 1.25 |
| goals_conceded | 0.728 | 0.914 | 6.66 | 6.64 | 4.72 | 7.09 |

Sampled goal bars sitting on a 0-minute column: 0.069.

MAE of the per-card remainder sum (sample mean vs actual, baseline vs actual):

| channel | GRU sample MAE | baseline MAE |
|---|---:|---:|
| goals | 0.50 | 0.48 |
| assists | 0.51 | 0.45 |
| clean_sheets | 0.70 | 0.55 |
| goals_conceded | 2.15 | 1.64 |

## Goal-sum quintiles on the test remainder

Players ordered by mean sampled remainder goal height. Five equal groups.

| group | GRU sample | GRU actual | baseline expected | baseline actual |
|---|---:|---:|---:|---:|
| 1 | 0.00 | 0.01 | 0.01 | 0.00 |
| 2 | 0.04 | 0.02 | 0.01 | 0.00 |
| 3 | 0.18 | 0.06 | 0.13 | 0.09 |
| 4 | 0.53 | 0.46 | 0.67 | 0.49 |
| 5 | 2.47 | 2.46 | 2.52 | 2.43 |

## One card

Morgan Gibbs-White, MID, Nott'm Forest. Cut after fixture column 19. Chosen as the 2025-26 card with the most goal bars on the remainder, so the spikes are visible.
The sample is the first draw, not a selected draw.

Plot: `data/plots/flashcard_gru_trial.png`.

## Read

The greedy goal chart is empty: mean argmax sum 0.00 against an actual 0.60.
16 draws average 0.64 goal bars.
Played-week teacher-forced goal NLL is 0.320 against the baseline 0.321.
The baseline is ahead on clean sheets, goals conceded, and the MAE of every remainder sum.
Blank-token probabilities are not the sampler. On a played week they put probability 0.66 on a goal bar whose true rate is 0.10.
