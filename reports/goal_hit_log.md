# Goal-hit log

The score is the share of a player's hidden goal weeks that land in the top 10% of that player's own played weeks. A played week has at least 60 minutes. A player counts with at least one hidden goal and eight played hidden weeks. The rank uses the probability the goal bar is at least 1. Ties are broken at random and the score is the mean of 20 draws. Chance is 10%.

Compare a new row with earlier rows on the same exam. A higher top-10% score on that exam is an improvement.

Train seasons for every row so far: 2022-23, 2023-24, 2024-25. Test season: 2025-26.

| exam | what is held fixed |
|---|---|
| player card | One player. Cut after the first 50% of that player's fixture columns. 129 players. |
| squad card | One club. Cut after gameweek 19 for every club. 135 players. |

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

Best on the player card: minutes, 11.4%. Best on the squad card: squad vision with the marked chart, 13.1%.

The marked chart adds a position color, a solid band at 60 minutes, and an eight-pixel opponent conceded bar. The top 10% rose from 12.2% to 13.1%. The top 20% fell from 23.5% to 21.6%. Expected goals on played weeks were 28.1 against 21.4 actual.

Add the next model as one row. Keep the exam name when the card, the cut, and the player rule stay the same. Give a changed setup a new exam name.
