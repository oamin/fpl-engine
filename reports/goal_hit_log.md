# Goal-hit log

The score is the share of a player's hidden goal weeks that land in the top 10% of that player's own weeks. The rank uses the probability the goal bar is at least 1. Ties are broken at random and the score is the mean of 20 draws. Chance is 10%. Which weeks count is part of the exam: the squad-card rows count a week at 60 minutes or more, and the all-minutes rows count every recorded appearance.

Compare a new row with earlier rows on the same exam. A higher top-10% score on that exam is an improvement.

Train seasons for every row so far: 2022-23, 2023-24, 2024-25. Test season: 2025-26.

| exam | what is held fixed |
|---|---|
| player card | One player. Cut after the first 50% of that player's fixture columns. 129 players. |
| squad card | One club. Cut after gameweek 19 for every club. A counted week has at least 60 minutes. 135 players. |
| squad card, all minutes | Same cut. Every recorded appearance counts, including a zero-minute bench row. 197 players. |

The flash-card GRU was trained on the player card and scored on log-loss and season totals. This percentage was not computed for it.

| logged (UK) | model | exam | top 10% | top 20% | top 50% |
|---|---|---|---:|---:|---:|
| 6 Nov 2026, 21:38 | column vision | player card | 11.0% | 21.9% | 53.6% |
| 6 Nov 2026, 21:38 | shrunk prefix rate | player card | 10.1% | 20.5% | 49.8% |
| 6 Nov 2026, 21:38 | minutes | player card | 11.4% | 21.7% | 52.6% |
| 7 Nov 2026, 18:32 | squad sequence | squad card | 8.6% | 17.9% | 47.7% |
| 7 Nov 2026, 18:32 | squad vision | squad card | 12.2% | 23.5% | 53.8% |
| 7 Nov 2026, 19:55 | squad vision, marked chart | squad card | 13.1% | 21.6% | 54.4% |
| 7 Nov 2026, 18:32 | prefix rate, strength 8 | squad card | 10.9% | 20.0% | 50.6% |
| 7 Nov 2026, 18:32 | group rate | squad card | 10.5% | 20.4% | 49.8% |
| 7 Nov 2026, 18:32 | minutes | squad card | 11.3% | 22.3% | 52.6% |
| 7 Nov 2026, 20:27 | squad sequence | squad card, all minutes | 15.8% | 31.2% | 62.8% |
| 7 Nov 2026, 20:27 | squad vision | squad card, all minutes | 22.3% | 40.4% | 75.8% |
| 7 Nov 2026, 20:27 | prefix rate, strength 8 | squad card, all minutes | 19.3% | 37.4% | 75.8% |
| 7 Nov 2026, 20:27 | group rate | squad card, all minutes | 20.0% | 37.8% | 75.8% |
| 7 Nov 2026, 20:27 | minutes | squad card, all minutes | 21.9% | 38.4% | 77.9% |

Best on the player card: minutes, 11.4%. Best on the squad card: squad vision with the marked chart, 13.1%. Best on the squad card with every appearance counted: squad vision, 22.3%.

The marked chart adds a position color, a solid band at 60 minutes, and an eight-pixel opponent conceded bar. The top 10% rose from 12.2% to 13.1%. The top 20% fell from 23.5% to 21.6%. Expected goals on played weeks were 28.1 against 21.4 actual.

The all-minutes rows drop that 60-minute rule. The readers take the minute count itself, and a bench row stays in the ranking. Sorting by minutes catches 21.9%. Squad vision catches 22.3%.

Add the next model as one row. Keep the exam name when the card, the cut, and the player rule stay the same. Give a changed setup a new exam name.

## Bar correspondence

A bar is an appearance with at least one goal or assist. The window is every club in 2025-26 from gameweek 20 to 38. Drawn bars are samples from the reader, averaged over 20 draws. Covered is the share of actual bars that land on the same appearance as a drawn bar. On a real week is the share of drawn bars that land on an actual bar.

| logged (UK) | reader | channel | actual bars | drawn bars | covered | on a real week |
|---|---|---|---:|---:|---:|---:|
| 7 Nov 2026, 23:15 | squad vision | goals | 459 | 620.2 | 11.3% | 8.4% |
| 7 Nov 2026, 23:15 | squad sequence | goals | 459 | 562.5 | 9.6% | 7.9% |
| 7 Nov 2026, 23:15 | prefix rate, strength 8 | goals | 459 | 489.8 | 17.0% | 15.9% |
| 7 Nov 2026, 23:15 | group rate | goals | 459 | 501.9 | 15.9% | 14.5% |
| 7 Nov 2026, 23:15 | same count, random weeks | goals | 459 | 459.0 | 20.4% | 20.4% |
| 7 Nov 2026, 23:15 | squad vision | assists | 445 | 590.8 | 9.4% | 7.1% |
| 7 Nov 2026, 23:15 | squad sequence | assists | 445 | 519.6 | 8.2% | 7.0% |
| 7 Nov 2026, 23:15 | prefix rate, strength 8 | assists | 445 | 443.6 | 11.6% | 11.6% |
| 7 Nov 2026, 23:15 | group rate | assists | 445 | 453.7 | 10.4% | 10.2% |
| 7 Nov 2026, 23:15 | same count, random weeks | assists | 445 | 445.0 | 15.1% | 15.1% |

The 23:33 run gives the goal and assist heads a learned position term, the player's own minutes and visible bars, and the match-up header. Nothing is removed for goalkeepers. Vision's probability of a goal bar on a hidden appearance is 5.4% for a goalkeeper and 7.2% for a forward. Drawn goal bars: 533.6 against 459 actual, 12.3% of actual bars covered, 10.6% of drawn bars on a real goal week.
