# Squad-card trial

Exploratory. One held-out season. Not a gated score comparison.
2026-27 was not read.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Cards: train 60, test 20. One card is a club-season.
Grid: 40 players by 40 fixtures. Cut after gameweek 19 for every club.
Minutes, venue, opponent goals conceded, and regulars-on-the-pitch stay visible.
Goals and assists after the cut are hidden. The segment is predicted in one pass.
Wall clock: 65 seconds on CPU.

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
