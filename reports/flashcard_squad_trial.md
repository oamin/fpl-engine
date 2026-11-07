# Squad-card trial

Exploratory. One held-out season. Not a gated score comparison.
2026-27 was not read.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Cards: train 60, test 20. One card is a club-season.
Grid: 40 players by 40 fixtures. Cut after gameweek 19 for every club.
Minutes, venue, opponent goals conceded, and regulars-on-the-pitch stay visible.
Goals and assists after the cut are hidden. The segment is predicted in one pass.
Wall clock: 66 seconds on CPU.

## Readers

Sequence: the minute count, presence, position, and the visible bars as tokens.
Vision: the same grid painted in pixels. Minutes use sixteen pixels, with no
60-minute step, plus a position color and an eight-pixel opponent conceded bar.
Both then attend across the 40 rows and across all 40 columns.
The personal baseline is the player's visible bars pulled toward the position-and-minutes rate
with strength 8. The group baseline is that rate alone.

## Hidden appearances

Every recorded appearance is included, at whatever minutes. Lower negative log-likelihood is better.

| reader | goal nll | assist nll | goal P | goal rate | expected goals | actual goals |
|---|---:|---:|---:|---:|---:|---:|
| sequence | 0.184 | 0.184 | 0.056 | 0.045 | 34.77 | 25.05 |
| vision | 0.181 | 0.176 | 0.062 | 0.045 | 39.24 | 25.05 |
| prefix | 0.157 | 0.159 | 0.048 | 0.045 | 27.39 | 25.05 |
| group | 0.153 | 0.156 | 0.050 | 0.045 | 28.20 | 25.05 |

## Where the goals land

Players with a goal and at least eight appearances after the cut: 197.
An appearance is any recorded minutes, including a zero-minute bench row.
A random appearance in that group is a goal 12.4% of the time.
The vision model's top 10% of appearances are goals 22.8% of the time.

| rank by | top 10% | top 20% | top 50% |
|---|---:|---:|---:|
| sequence | 0.158 | 0.312 | 0.628 |
| vision | 0.223 | 0.404 | 0.758 |
| prefix | 0.193 | 0.374 | 0.758 |
| group | 0.200 | 0.378 | 0.758 |
| minutes | 0.219 | 0.384 | 0.779 |
| chance | 0.100 | 0.200 | 0.500 |

Plot: `data/plots/flashcard_squad_trial.png`.

## Bars across the hidden gameweeks

A bar is an appearance with height at least 1. Each reader draws one by sampling
its probabilities. The totals are every club, every appearance after the cut,
averaged over 20 draws. Covered is the share of actual bars that sit on the same
appearance as a drawn bar. On a real week is the share of drawn bars that sit on
an actual bar. Same count, random weeks keeps each player's bar count and shuffles
the appearances.

| reader | channel | actual bars | drawn bars | covered | on a real week |
|---|---|---:|---:|---:|---:|
| vision | goals | 459 | 620.2 | 11.3% | 8.4% |
| sequence | goals | 459 | 562.5 | 9.6% | 7.9% |
| prefix | goals | 459 | 489.8 | 17.0% | 15.9% |
| group | goals | 459 | 501.9 | 15.9% | 14.5% |
| random | goals | 459 | 459.0 | 20.4% | 20.4% |
| vision | assists | 445 | 590.8 | 9.4% | 7.1% |
| sequence | assists | 445 | 519.6 | 8.2% | 7.0% |
| prefix | assists | 445 | 443.6 | 11.6% | 11.6% |
| group | assists | 445 | 453.7 | 10.4% | 10.2% |
| random | assists | 445 | 445.0 | 15.1% | 15.1% |

One vision draw for Man Utd, gameweeks 20–38: `data/plots/flashcard_squad_bars.png`.
