# Squad-card trial

Exploratory. One held-out season. Not a gated score comparison.
2026-27 was not read.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Cards: train 60, test 20. One card is a club-season.
Grid: 40 players by 40 fixtures. Cut after gameweek 19 for every club.
Minutes, venue, opponent goals conceded, and regulars-on-the-pitch stay visible.
Goals and assists after the cut are hidden. The segment is predicted in one pass.
Wall clock: 72 seconds on CPU.

## Readers

Position, the player's own minutes and visible bars, and the match-up
each have a learned effect on the goal and assist heads.
A club's earlier-season goals per game, and the opponent's earlier-season
goals conceded per game, are shared across every card of that club.
A club with no earlier season, once other clubs have one, is the promoted case.
This season's own scoring rate and the opponent's conceding rate use only
fixtures already played, and a hidden column stops at the cut.
Learned vision probability of a goal bar on a hidden appearance: goalkeeper 0.1%, defender 2.4%, midfielder 5.4%, forward 11.2%.
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
| sequence | 0.158 | 0.162 | 0.048 | 0.045 | 27.08 | 25.05 |
| vision | 0.157 | 0.162 | 0.047 | 0.045 | 25.94 | 25.05 |
| prefix | 0.157 | 0.159 | 0.048 | 0.045 | 27.39 | 25.05 |
| group | 0.153 | 0.156 | 0.050 | 0.045 | 28.20 | 25.05 |

## Where the goals land

Players with a goal and at least eight appearances after the cut: 197.
An appearance is any recorded minutes, including a zero-minute bench row.
A random appearance in that group is a goal 12.4% of the time.
The vision model's top 10% of appearances are goals 23.1% of the time.

| rank by | top 10% | top 20% | top 50% |
|---|---:|---:|---:|
| sequence | 0.209 | 0.395 | 0.776 |
| vision | 0.238 | 0.419 | 0.795 |
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
| vision | goals | 459 | 466.2 | 15.1% | 14.8% |
| sequence | goals | 459 | 492.6 | 16.3% | 15.2% |
| prefix | goals | 459 | 489.8 | 17.0% | 15.9% |
| group | goals | 459 | 501.9 | 15.9% | 14.5% |
| random | goals | 459 | 459.0 | 20.4% | 20.4% |
| vision | assists | 445 | 406.0 | 10.2% | 11.2% |
| sequence | assists | 445 | 423.6 | 10.8% | 11.3% |
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
| 80% | vision | goals | 199 | 192.5 | 16.2% | 16.8% | 0.1% | 11.4% |
| 80% | vision | assists | 192 | 169.1 | 9.3% | 10.6% | 0.1% | 11.4% |
| 80% | sequence | goals | 199 | 218.2 | 16.3% | 14.9% | 0.1% | 12.0% |
| 80% | sequence | assists | 192 | 187.0 | 10.6% | 10.9% | 0.1% | 12.0% |
| 80% | prefix | goals | 199 | 202.7 | 17.9% | 17.6% | 0.1% | 12.9% |
| 80% | prefix | assists | 192 | 192.2 | 12.3% | 12.3% | 0.1% | 12.9% |
| 60% | vision | goals | 360 | 372.9 | 15.6% | 15.0% | 0.1% | 11.1% |
| 60% | vision | assists | 351 | 323.7 | 10.6% | 11.5% | 0.1% | 11.1% |
| 60% | sequence | goals | 360 | 394.6 | 16.5% | 15.1% | 0.1% | 11.3% |
| 60% | sequence | assists | 351 | 350.5 | 12.1% | 12.1% | 0.1% | 11.3% |
| 60% | prefix | goals | 360 | 384.6 | 16.0% | 15.0% | 0.1% | 13.2% |
| 60% | prefix | assists | 351 | 366.5 | 12.4% | 11.8% | 0.1% | 13.2% |
| 40% | vision | goals | 544 | 568.6 | 15.9% | 15.2% | 0.1% | 11.3% |
| 40% | vision | assists | 531 | 514.4 | 10.5% | 10.9% | 0.1% | 11.3% |
| 40% | sequence | goals | 544 | 570.4 | 15.8% | 15.1% | 0.1% | 10.9% |
| 40% | sequence | assists | 531 | 520.0 | 10.8% | 11.0% | 0.1% | 10.9% |
| 40% | prefix | goals | 544 | 602.1 | 16.8% | 15.2% | 0.1% | 13.2% |
| 40% | prefix | assists | 531 | 552.1 | 11.8% | 11.3% | 0.1% | 13.2% |
| 20% | vision | goals | 730 | 729.0 | 15.2% | 15.2% | 0.1% | 10.9% |
| 20% | vision | assists | 707 | 639.8 | 10.7% | 11.8% | 0.1% | 10.9% |
| 20% | sequence | goals | 730 | 697.5 | 14.3% | 14.9% | 0.1% | 9.9% |
| 20% | sequence | assists | 707 | 619.1 | 9.8% | 11.2% | 0.1% | 9.9% |
| 20% | prefix | goals | 730 | 775.8 | 16.0% | 15.1% | 0.1% | 13.4% |
| 20% | prefix | assists | 707 | 704.7 | 10.9% | 11.0% | 0.1% | 13.4% |

Four vision draws for Man Utd: `data/plots/flashcard_squad_reveals.png`.
The title on each panel is that one draw for the whole club.

## Club and match-up

Vision's goal-bar rate for forwards on the hidden remainder. The club and
match-up shift is learned separately for each position. Promoted clubs
are the sides missing from 2024-25.
A forward at a club that has been scoring, against a side that has been conceding,
is where that shift should show.

| revealed | group | prior goals/game | actual bars | drawn bars | forward goal rate |
|---|---|---:|---:|---:|---:|
| 80% | Man Utd | 1.36 | 14 | 10.9 | 7.0% |
| 80% | established | 1.53 | 172 | 171.1 | 12.2% |
| 80% | promoted | 1.12 | 27 | 26.3 | 7.4% |
| 80% | Man Utd vs promoted |  |  |  | 6.5% |
| 80% | Man Utd vs established |  |  |  | 7.1% |
| 60% | Man Utd | 1.36 | 27 | 20.6 | 6.1% |
| 60% | established | 1.53 | 317 | 323.9 | 11.9% |
| 60% | promoted | 1.12 | 43 | 47.2 | 7.3% |
| 60% | Man Utd vs promoted |  |  |  | 6.4% |
| 60% | Man Utd vs established |  |  |  | 6.0% |
| 40% | Man Utd | 1.36 | 41 | 29.8 | 7.1% |
| 40% | established | 1.53 | 478 | 493.3 | 12.2% |
| 40% | promoted | 1.12 | 66 | 74.0 | 7.4% |
| 40% | Man Utd vs promoted |  |  |  | 6.3% |
| 40% | Man Utd vs established |  |  |  | 7.2% |
| 20% | Man Utd | 1.36 | 54 | 37.5 | 7.1% |
| 20% | established | 1.53 | 637 | 626.8 | 11.8% |
| 20% | promoted | 1.12 | 93 | 91.3 | 6.8% |
| 20% | Man Utd vs promoted |  |  |  | 5.2% |
| 20% | Man Utd vs established |  |  |  | 7.2% |

Promoted sides in 2025-26, absent from the previous season: Leeds, Burnley, Sunderland.

Forwards at 40% revealed, one club at a time.

| club | prior goals/game | actual bars | drawn bars | forward goal rate |
|---|---:|---:|---:|---:|
| Man City | 2.25 | 38 | 33.4 | 37.3% |
| Arsenal | 2.07 | 34 | 37.7 | 9.9% |
| Liverpool | 2.07 | 35 | 43.9 | 13.6% |
| Man Utd | 1.36 | 41 | 29.0 | 7.1% |
| Leeds | 1.18 | 26 | 24.9 | 8.5% |
| Burnley | 1.05 | 19 | 23.4 | 5.0% |
| Sunderland | 0.00 | 21 | 21.6 | 10.2% |
