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
| 7 Nov 2026, 23:42 | squad sequence, position-rate head | squad card, all minutes | 21.2% | 39.3% | 77.3% |
| 7 Nov 2026, 23:42 | squad vision, position-rate head | squad card, all minutes | 22.8% | 42.6% | 80.5% |
| 7 Nov 2026, 23:42 | prefix rate, strength 8 | squad card, all minutes | 19.3% | 37.4% | 75.8% |
| 7 Nov 2026, 23:42 | group rate | squad card, all minutes | 20.0% | 37.8% | 75.8% |

Best on the player card: minutes, 11.4%. Best on the squad card: squad vision with the marked chart, 13.1%. Best on the squad card with every appearance counted: squad vision with the position-rate head, 22.8%.

The marked chart adds a position color, a solid band at 60 minutes, and an eight-pixel opponent conceded bar. The top 10% rose from 12.2% to 13.1%. The top 20% fell from 23.5% to 21.6%. Expected goals on played weeks were 28.1 against 21.4 actual.

The all-minutes rows drop that 60-minute rule. The readers take the minute count itself, and a bench row stays in the ranking. Sorting by minutes catches 21.9%. Squad vision at 20:27 catches 22.3%. The 23:42 position-rate head catches 22.8%, and the sequence reader on that same exam rises from 15.8% to 21.2%.

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

## Remainders from gameweek 1

The 23:42 run changes the fit. The shared goal and assist heads start at zero, so a position-by-minutes table has to explain the labels, and those position parameters take ten times the learning rate. Each training card shows a random 20–80% prefix starting at gameweek 1. Nothing is blanked for a goalkeeper.

After training, every 2025-26 card is shown from gameweek 1 through 80%, 60%, 40%, or 20% of its fixtures. The reader draws the hidden remainder. Totals are 20 draws. GK and FWD are the probability of a goal bar on a hidden appearance.

| logged (UK) | revealed | reader | channel | actual bars | drawn bars | covered | on a real week | GK | FWD |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 7 Nov 2026, 23:42 | 80% | squad vision | goals | 199 | 184.9 | 15.4% | 16.5% | 0.5% | 11.6% |
| 7 Nov 2026, 23:42 | 80% | squad sequence | goals | 199 | 228.8 | 15.7% | 13.6% | 0.3% | 12.7% |
| 7 Nov 2026, 23:42 | 80% | prefix rate, strength 8 | goals | 199 | 202.7 | 17.9% | 17.6% | 0.1% | 12.9% |
| 7 Nov 2026, 23:42 | 60% | squad vision | goals | 360 | 354.9 | 14.4% | 14.6% | 0.5% | 11.4% |
| 7 Nov 2026, 23:42 | 60% | squad sequence | goals | 360 | 402.9 | 15.3% | 13.7% | 0.3% | 11.8% |
| 7 Nov 2026, 23:42 | 60% | prefix rate, strength 8 | goals | 360 | 384.6 | 16.0% | 15.0% | 0.1% | 13.2% |
| 7 Nov 2026, 23:42 | 40% | squad vision | goals | 544 | 538.4 | 14.6% | 14.7% | 0.5% | 11.4% |
| 7 Nov 2026, 23:42 | 40% | squad sequence | goals | 544 | 557.3 | 13.8% | 13.5% | 0.3% | 10.9% |
| 7 Nov 2026, 23:42 | 40% | prefix rate, strength 8 | goals | 544 | 602.1 | 16.8% | 15.2% | 0.1% | 13.2% |
| 7 Nov 2026, 23:42 | 20% | squad vision | goals | 730 | 688.9 | 14.4% | 15.3% | 0.5% | 11.2% |
| 7 Nov 2026, 23:42 | 20% | squad sequence | goals | 730 | 658.3 | 11.6% | 12.9% | 0.3% | 9.7% |
| 7 Nov 2026, 23:42 | 20% | prefix rate, strength 8 | goals | 730 | 775.8 | 16.0% | 15.1% | 0.1% | 13.4% |

On the gameweek-19 exam the same vision model puts a goal bar at 0.5% for a goalkeeper, 1.9% for a defender, 5.1% for a midfielder, and 11.5% for a forward. Drawn goal bars on gameweeks 20–38: 445.8 against 459 actual, 14.1% covered, 14.6% on a real week. A shuffle of the correct count still covers 20.4%. Assist remainders stay thinner: at 20% revealed, vision draws 614.9 assist bars against 707 actual and covers 9.9%.
