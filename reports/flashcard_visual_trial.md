# Flash-card vision trial

Exploratory. One held-out season. Not a gated score comparison.
2026-27 was not read.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Cards with at least 12 fixtures after dropping repeated log lines: train 2348, test 818.
Longest card: 39 fixtures. The grid holds 40.
Test cut: first 50% of fixture columns visible.
Input picture: 400×75. Minutes, home/away, and opponent stay visible.
Hidden event cells are flat gray. Gray is not a zero.
Wall clock: 11 seconds on CPU.

## Fit

Each of the 40 columns is one visual token, read from its own pixels.
Two transformer layers, hidden 64. Four softmax heads.
Cross-entropy on the hidden columns. Prefix fraction during training is uniform between 30% and 70%.
Epoch mean NLL per hidden bar: 0.341, 0.238, 0.233, 0.237, 0.236, 0.235, 0.226, 0.228, 0.224, 0.218.

Baseline: the player's prefix histogram, shrunk toward a position × minutes-bin
prior fit on the training seasons only. The baseline sees position. The picture does not.

## Held-out negative log-likelihood

Lower is better. Played columns are remainder fixtures with at least 60 minutes.

| model | goals | assists | clean sheets | goals conceded |
|---|---:|---:|---:|---:|
| Vision, all columns | 0.106 | 0.110 | 0.162 | 0.453 |
| Vision, played | 0.337 | 0.350 | 0.626 | 1.363 |
| Baseline, all columns | 0.102 | 0.104 | 0.151 | 0.448 |
| Baseline, played | 0.321 | 0.329 | 0.585 | 1.330 |

## Remainder sums

Mean actual remainder sum: goals 0.599, assists 0.559, clean_sheets 1.269, goals_conceded 6.659.
Mean vision expected sum: goals 0.606, assists 0.444, clean_sheets 0.791, goals_conceded 6.938.
Mean vision argmax sum: goals 0.004, assists 0.000, clean_sheets 0.001, goals_conceded 6.582.
Mean baseline expected sum: goals 0.664, assists 0.596, clean_sheets 1.247, goals_conceded 7.086.
MAE of the per-card remainder sum, vision: goals 0.506, assists 0.458, clean_sheets 0.683, goals_conceded 1.670.
MAE of the per-card remainder sum, baseline: goals 0.478, assists 0.446, clean_sheets 0.552, goals_conceded 1.641.

Played-week rate of a bar of at least 1, then the vision model's mean probability of that:

| channel | actual rate | vision P |
|---|---:|---:|
| goals | 0.099 | 0.107 |
| assists | 0.097 | 0.070 |
| clean_sheets | 0.272 | 0.166 |
| goals_conceded | 0.728 | 0.725 |

## Which played week gets the goal

Players with at least one goal week and eight played weeks in the remainder: 129.
The vision score is the probability the goal bar is at least 1. Ties are broken at random, averaged over 20 draws.

| rank by | top 10% | top 20% | top 30% | top 40% | top 50% |
|---|---:|---:|---:|---:|---:|
| vision | 0.110 | 0.219 | 0.335 | 0.441 | 0.536 |
| baseline | 0.101 | 0.205 | 0.297 | 0.402 | 0.498 |
| minutes | 0.114 | 0.217 | 0.311 | 0.426 | 0.526 |
| chance | 0.100 | 0.200 | 0.300 | 0.400 | 0.500 |

## One card

Morgan Gibbs-White. Cut after fixture column 19.
The middle picture is the most likely height in each hidden cell, not a sample.

Plot: `data/plots/flashcard_visual_card.png`.
Capture curve: `data/plots/flashcard_visual_capture.png`.
