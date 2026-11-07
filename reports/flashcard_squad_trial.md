# Squad-card trial

Exploratory. One held-out season. Not a gated score comparison.
2026-27 was not read.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Cards: train 60, test 20. One card is a club-season.
Grid: 40 players by 40 fixtures. Cut after gameweek 19 for every club.
Minutes, venue, opponent goals conceded, and regulars-on-the-pitch stay visible.
Goals and assists after the cut are hidden. The segment is predicted in one pass.
Wall clock: 52 seconds on CPU.

## Readers

Sequence: minutes, the 60-minute flag, position, and the visible bars as tokens.
Vision: the same grid painted in pixels. A shared cell encoder reads each patch.
Both then attend across the 40 rows and across all 40 columns.
The personal baseline is the player's visible bars pulled toward the position-and-minutes rate
with strength 8. The group baseline is that rate alone.

## Played hidden weeks

A played week has at least 60 minutes. Lower negative log-likelihood is better.

| reader | goal nll | assist nll | goal P | goal rate | expected goals | actual goals |
|---|---:|---:|---:|---:|---:|---:|
| sequence | 0.338 | 0.352 | 0.072 | 0.099 | 17.50 | 21.35 |
| vision | 0.363 | 0.346 | 0.106 | 0.099 | 23.55 | 21.35 |
| prefix | 0.328 | 0.330 | 0.104 | 0.099 | 23.04 | 21.35 |
| group | 0.321 | 0.328 | 0.107 | 0.099 | 23.69 | 21.35 |

## Where the goals land

Players with a goal and at least eight played weeks after the cut: 135.

| rank by | top 10% | top 20% | top 50% |
|---|---:|---:|---:|
| sequence | 0.086 | 0.179 | 0.477 |
| vision | 0.122 | 0.235 | 0.538 |
| prefix | 0.109 | 0.200 | 0.506 |
| group | 0.105 | 0.204 | 0.498 |
| minutes | 0.113 | 0.223 | 0.526 |
| chance | 0.100 | 0.200 | 0.500 |

Plot: `data/plots/flashcard_squad_trial.png`.
