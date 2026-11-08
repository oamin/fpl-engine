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
| 8 Nov 2026, 00:16 | squad sequence, club and match-up | squad card, all minutes | 20.9% | 39.5% | 77.6% |
| 8 Nov 2026, 00:16 | squad vision, club and match-up | squad card, all minutes | 23.8% | 41.9% | 79.5% |
| 8 Nov 2026, 15:17 | squad sequence, squad coupling | squad card, all minutes | 20.6% | 39.3% | 77.6% |
| 8 Nov 2026, 15:17 | squad vision, squad coupling | squad card, all minutes | 24.7% | 41.5% | 80.1% |
| 8 Nov 2026, 16:08 | squad sequence, opponent form | squad card, all minutes | 20.3% | 39.1% | 78.4% |
| 8 Nov 2026, 16:08 | squad vision, opponent form | squad card, all minutes | 21.5% | 41.1% | 78.9% |
| 8 Nov 2026, 17:15 | squad sequence, paired gameweeks | squad card, all minutes | 18.1% | 35.1% | 71.1% |
| 8 Nov 2026, 18:14 | squad sequence, opponent form | squad card, all minutes | 20.3% | 39.1% | 78.4% |
| 8 Nov 2026, 18:14 | squad vision, player rows only | squad card, all minutes | 20.6% | 39.4% | 78.8% |

Best on the player card: minutes, 11.4%. Best on the squad card: squad vision with the marked chart, 13.1%. Best on the squad card with every appearance counted: squad vision with squad coupling, 24.7%.

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

## Club and match-up

The 00:16 run on 8 Nov gives each position its own weight on the club and on the match-up. The match-up is the club's goals per game in earlier seasons, the opponent's goals conceded per game in earlier seasons, and both clubs' rates so far this season, using only fixtures already played. A club missing from 2024-25 is the promoted case: Leeds, Burnley, and Sunderland. Sunderland has no earlier season in the file.

Vision, forwards, 40% revealed from gameweek 1:

| logged (UK) | club | prior goals/game | actual bars | drawn bars | forward goal rate |
|---|---|---:|---:|---:|---:|
| 8 Nov 2026, 00:16 | Man City | 2.25 | 38 | 33.4 | 37.3% |
| 8 Nov 2026, 00:16 | Liverpool | 2.07 | 35 | 43.9 | 13.6% |
| 8 Nov 2026, 00:16 | Arsenal | 2.07 | 34 | 37.7 | 9.9% |
| 8 Nov 2026, 00:16 | Man Utd | 1.36 | 41 | 29.0 | 7.1% |
| 8 Nov 2026, 00:16 | Leeds | 1.18 | 26 | 24.9 | 8.5% |
| 8 Nov 2026, 00:16 | Burnley | 1.05 | 19 | 23.4 | 5.0% |
| 8 Nov 2026, 00:16 | Sunderland | 0.00 | 21 | 21.6 | 10.2% |

Man Utd's whole-club remainder, vision, mean of 20 draws: at 80% revealed, 14 real and 10.9 drawn; at 40%, 41 real and 29.8 drawn; at 20%, 54 real and 37.5 drawn. Man Utd forwards against a promoted opponent are at 6.3%. Against an established opponent they are at 7.2%.

## Squad coupling

The 15:17 run on 8 Nov keeps the club and match-up head and adds the other squad at each fixture. The reader sees that squad's positions, minutes, and the goals and assists already shown. The same cut masks both sides. A hidden goal on the other card does not enter the input.

On the all-minutes exam the vision top 10% is 24.7% and the sequence top 10% is 20.6%. Goal-bar overlap on gameweeks 20–38 is 14.9% covered for vision, against 20.4% for a same-count random draw. Man Utd forwards at 40% revealed stay at 7.1%. Against a promoted opponent they are at 6.4%.

The 16:08 run adds the opponent's goals scored and conceded over the previous five games, and the minutes of their leading scorer over the previous three. A hidden column stops at the cut. Vision's top 10% on the same exam is 21.5%.

## Paired gameweeks

The 17:15 run on 8 Nov trains the sequence reader on a pair of clubs. Each column reads the other squad at that gameweek and the two gameweeks before it. Minutes stay visible. Goals and assists after the shared cut stay masked on both cards. Vision is left as it was.

On the all-minutes exam the top 10% is 18.1%. Goal remainders from gameweek 1: at 80% revealed, 199 actual and 238.8 drawn, 11.3% covered, 9.4% on a real week; at 20% revealed, 730 actual and 886.4 drawn, 11.2% covered, 9.2% on a real week. A goalkeeper's hidden appearance is a goal bar at 1.9% when 80% is revealed and at 1.8% when 20% is revealed. A forward is at 12.9% and 12.7%.

Assist remainders: at 80% revealed, 192 actual and 223.4 drawn, 8.3% covered, 7.1% on a real week; at 20% revealed, 707 actual and 823.9 drawn, 7.9% covered, 6.8% on a real week.

## Player rows only

The 18:14 run on 8 Nov uses the 16:08 training procedure and removes the five header rows from the picture: venue, opponent goals conceded, regulars, earlier-season goals per game, and earlier-season goals conceded per game. Each player row is minutes, goals, assists, and a position color. The numeric match-up and the other squad stay. Sequence is unchanged, and its top 10% reproduces at 20.3%.

Vision's top 10% on the all-minutes exam is 20.6%. With the header still painted, the same procedure scored 21.5%. The best line on this exam stays the 15:17 squad-coupling vision at 24.7%. Goal remainders from gameweek 1, vision: at 80% revealed, 199 actual and 202.9 drawn, 16.6% covered, 16.3% on a real week; at 20% revealed, 730 actual and 778.2 drawn, 16.1% covered, 15.1% on a real week. A goalkeeper is at 0.1% and a forward at 11.7%.
