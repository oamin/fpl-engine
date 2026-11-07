# Squad-card trial

Exploratory. One held-out season. Not a gated score comparison.
2026-27 was not read.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Cards: train 60, test 20. One card is a club-season.
Grid: 40 players by 40 fixtures. Cut after gameweek 19 for every club.
Minutes, venue, opponent goals conceded, and regulars-on-the-pitch stay visible.
Goals and assists after the cut are hidden. The segment is predicted in one pass.
Wall clock: 70 seconds on CPU.

## Readers

Position, the player's own minutes and visible bars, and the match-up header
each have a learned effect on the goal and assist heads.
Learned vision probability of a goal bar on a hidden appearance: goalkeeper 0.5%, defender 1.9%, midfielder 5.1%, forward 11.5%.
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
| sequence | 0.159 | 0.162 | 0.048 | 0.045 | 27.84 | 25.05 |
| vision | 0.156 | 0.159 | 0.045 | 0.045 | 24.93 | 25.05 |
| prefix | 0.157 | 0.159 | 0.048 | 0.045 | 27.39 | 25.05 |
| group | 0.153 | 0.156 | 0.050 | 0.045 | 28.20 | 25.05 |

## Where the goals land

Players with a goal and at least eight appearances after the cut: 197.
An appearance is any recorded minutes, including a zero-minute bench row.
A random appearance in that group is a goal 12.4% of the time.
The vision model's top 10% of appearances are goals 22.6% of the time.

| rank by | top 10% | top 20% | top 50% |
|---|---:|---:|---:|
| sequence | 0.212 | 0.393 | 0.773 |
| vision | 0.228 | 0.426 | 0.805 |
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
| vision | goals | 459 | 445.8 | 14.1% | 14.6% |
| sequence | goals | 459 | 492.1 | 14.6% | 13.7% |
| prefix | goals | 459 | 489.8 | 17.0% | 15.9% |
| group | goals | 459 | 501.9 | 15.9% | 14.5% |
| random | goals | 459 | 459.0 | 20.4% | 20.4% |
| vision | assists | 445 | 396.3 | 9.5% | 10.6% |
| sequence | assists | 445 | 439.6 | 9.8% | 9.9% |
| prefix | assists | 445 | 443.6 | 11.6% | 11.6% |
| group | assists | 445 | 453.7 | 10.4% | 10.2% |
| random | assists | 445 | 445.0 | 15.1% | 15.1% |

One vision draw for Man Utd, gameweeks 20–38: `data/plots/flashcard_squad_bars.png`.

## Remainders from gameweek 1

Each card is shown from gameweek 1 through 80%, 60%, 40%, or 20% of its fixtures.
The reader then draws the hidden remainder. Covered is the share of actual remainder
bars that sit on the same appearance as a drawn bar. On a real week is the share of
drawn bars that sit on an actual bar. The goalkeeper and forward columns are the
model's probability of a goal bar on the hidden appearances.

| revealed | reader | channel | actual bars | drawn bars | covered | on a real week | GK | FWD |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 80% | vision | goals | 199 | 184.9 | 15.4% | 16.5% | 0.5% | 11.6% |
| 80% | vision | assists | 192 | 165.6 | 9.4% | 10.9% | 0.5% | 11.6% |
| 80% | sequence | goals | 199 | 228.8 | 15.7% | 13.6% | 0.3% | 12.7% |
| 80% | sequence | assists | 192 | 205.8 | 10.6% | 9.9% | 0.3% | 12.7% |
| 80% | prefix | goals | 199 | 202.7 | 17.9% | 17.6% | 0.1% | 12.9% |
| 80% | prefix | assists | 192 | 192.2 | 12.3% | 12.3% | 0.1% | 12.9% |
| 60% | vision | goals | 360 | 354.9 | 14.4% | 14.6% | 0.5% | 11.4% |
| 60% | vision | assists | 351 | 312.6 | 9.9% | 11.2% | 0.5% | 11.4% |
| 60% | sequence | goals | 360 | 402.9 | 15.3% | 13.7% | 0.3% | 11.8% |
| 60% | sequence | assists | 351 | 372.3 | 11.6% | 10.9% | 0.3% | 11.8% |
| 60% | prefix | goals | 360 | 384.6 | 16.0% | 15.0% | 0.1% | 13.2% |
| 60% | prefix | assists | 351 | 366.5 | 12.4% | 11.8% | 0.1% | 13.2% |
| 40% | vision | goals | 544 | 538.4 | 14.6% | 14.7% | 0.5% | 11.4% |
| 40% | vision | assists | 531 | 489.8 | 9.7% | 10.5% | 0.5% | 11.4% |
| 40% | sequence | goals | 544 | 557.3 | 13.8% | 13.5% | 0.3% | 10.9% |
| 40% | sequence | assists | 531 | 530.7 | 10.0% | 10.0% | 0.3% | 10.9% |
| 40% | prefix | goals | 544 | 602.1 | 16.8% | 15.2% | 0.1% | 13.2% |
| 40% | prefix | assists | 531 | 552.1 | 11.8% | 11.3% | 0.1% | 13.2% |
| 20% | vision | goals | 730 | 688.9 | 14.4% | 15.3% | 0.5% | 11.2% |
| 20% | vision | assists | 707 | 614.9 | 9.9% | 11.4% | 0.5% | 11.2% |
| 20% | sequence | goals | 730 | 658.3 | 11.6% | 12.9% | 0.3% | 9.7% |
| 20% | sequence | assists | 707 | 602.9 | 8.0% | 9.4% | 0.3% | 9.7% |
| 20% | prefix | goals | 730 | 775.8 | 16.0% | 15.1% | 0.1% | 13.4% |
| 20% | prefix | assists | 707 | 704.7 | 10.9% | 11.0% | 0.1% | 13.4% |

Four vision draws for Man Utd: `data/plots/flashcard_squad_reveals.png`.
