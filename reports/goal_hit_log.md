# Goal-hit log

The score is the painted card against the actual card. Training and test cuts are 20%, 40%, 60%, and 80% of each club's fixtures, starting from gameweek 1. Every club and every row is included. The numbers are totals.

The reader is squad vision with squad coupling on the stripped card. Venue stays. Concede, regulars, own history, and opponent history are absent from the picture and from the match-up. The recent scoring rate stays. There is no odds bar. In training, each card draws one of those four cuts at random. The test scores all four. Each future fixture is given a number of bars equal to the sum of its bar probabilities. Those bars go to the rows with minutes and the highest probabilities. A bar matches a goal or an assist on the same row only. Within one gameweek includes the same appearance.

This table is the gate, logged at 15:53 UK on 9 Nov 2026. Test season 2025-26. The goal and assist bands in that run are eight pixels. A later pass paints those bands at sixteen pixels and does not replace this table.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 746 | 190 | 258 | 382 |
| 20% | assists | 707 | 681 | 103 | 183 | 221 |
| 40% | goals | 544 | 576 | 150 | 201 | 295 |
| 40% | assists | 531 | 523 | 74 | 134 | 156 |
| 60% | goals | 360 | 377 | 99 | 132 | 189 |
| 60% | assists | 351 | 341 | 45 | 87 | 94 |
| 80% | goals | 199 | 201 | 56 | 70 | 97 |
| 80% | assists | 192 | 182 | 24 | 49 | 52 |

A painted bar can sit next to a goal that another painted bar already covers, so the painted-within column is larger than the actual-within column. The gameweek-19 cut used earlier is not this score.

The previous gate, logged at 13:26 UK, kept concede, regulars, own history, and opponent history.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 738 | 192 | 257 | 382 |
| 20% | assists | 707 | 659 | 98 | 175 | 216 |
| 40% | goals | 544 | 584 | 153 | 204 | 303 |
| 40% | assists | 531 | 517 | 67 | 129 | 150 |
| 60% | goals | 360 | 381 | 97 | 134 | 192 |
| 60% | assists | 351 | 339 | 44 | 86 | 92 |
| 80% | goals | 199 | 201 | 56 | 72 | 99 |
| 80% | assists | 192 | 178 | 19 | 43 | 45 |

## One-gameweek step

The 14:00 run on 9 Nov uses the same reader. Each step reveals the true card through the previous gameweek, greys the next gameweek except player minutes, and chops every later column. The counts below use the same hidden fixtures as the gate.

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 521 | 137 | 182 | 276 |
| 20% | assists | 707 | 461 | 55 | 103 | 126 |
| 40% | goals | 544 | 403 | 100 | 138 | 211 |
| 40% | assists | 531 | 359 | 43 | 80 | 100 |
| 60% | goals | 360 | 263 | 68 | 90 | 134 |
| 60% | assists | 351 | 241 | 29 | 54 | 62 |
| 80% | goals | 199 | 137 | 35 | 48 | 66 |
| 80% | assists | 192 | 125 | 20 | 35 | 37 |

A rerun of the gate on the same weights matches the table above, except the 40% assist same-appearance count, which is 68 on that rerun. At 20% revealed the step puts 137 goal bars on the same appearance, against 192 for the gate, and 182 actual goal bars within one gameweek, against 257. The same gap shows at 40%, 60%, and 80%. The gate stays the one-shot card.

## Stepped training and match odds

The 14:30 run on 9 Nov trains the next-gameweek task. Each example shows the true past, greys the next gameweek, and chops every later column. The loss is only that gameweek. A second reader adds an odds row. A third reader is the full painter with the same odds row. The stepped rows without an odds bar are from the 15:00 UK pass, which keeps the venue band on the greyed gameweek. The odds rows are from the 15:29 UK pass, with the bar in blue, yellow, and purple.

The bar is opening home, draw, and away prices after the Shin de-vig. Blue is home, yellow is draw, purple is away, from the top of the row. The stepped reader sees that bar on known weeks and the step week. The full painter sees it on every known week and every grey week. The row is 8 pixels tall, so a 89/9/2 price is painted as 7 blue pixels and 1 yellow pixel.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | stepped | goals | 730 | 692 | 183 | 233 | 371 |
| 20% | stepped odds | goals | 730 | 677 | 184 | 231 | 373 |
| 20% | painter odds | goals | 730 | 764 | 193 | 259 | 390 |
| 80% | stepped | goals | 199 | 186 | 54 | 66 | 96 |
| 80% | stepped odds | goals | 199 | 184 | 54 | 67 | 97 |
| 80% | painter odds | goals | 199 | 207 | 56 | 73 | 101 |

At 20% revealed the stepped reader, with venue kept on the greyed week, puts 183 goal bars on the same appearance and 233 within one gameweek. With the blue, yellow, and purple odds bar it puts 184 and 231. The full painter puts 193 and 259, against 192 and 257 for the gate. At 40% and 60% the painter is still short of the gate. The gate stays the one-shot card without the odds bar.

## Header rows removed

The 15:53 UK pass on 9 Nov drops concede, regulars, own history, and opponent history from the picture and from the numeric match-up. Venue stays. The recent scoring rate stays.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | stripped | goals | 730 | 746 | 190 | 258 | 382 |
| 20% | stripped odds | goals | 730 | 744 | 192 | 259 | 385 |
| 80% | stripped | goals | 199 | 201 | 56 | 70 | 97 |
| 80% | stripped odds | goals | 199 | 201 | 57 | 72 | 99 |

The no-odds rows of this pass are the gate above. With the odds bar, at 20% revealed, the same card puts 192 goal bars on the same appearance and 259 within one gameweek.

## Stripped step

The 16:06 UK pass trains the next gameweek on the stripped card. Each example shows the true past, keeps venue and minutes on the next gameweek, and removes every later column. The loss is only that gameweek.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | stripped step | goals | 730 | 624 | 168 | 217 | 343 |
| 20% | stripped step | assists | 707 | 674 | 89 | 172 | 207 |
| 80% | stripped step | goals | 199 | 165 | 47 | 60 | 86 |
| 80% | stripped step | assists | 192 | 178 | 24 | 49 | 54 |

At 20% revealed the step puts 168 goal bars on the same appearance and 217 within one gameweek. The gate puts 190 and 258. The gate stays the one-shot stripped card.

## Taller event bands

The 21:54 UK pass on 9 Nov makes the goal band and the assist band sixteen pixels, the same height as the minute bar. One event fills five pixels, two fill eleven, and three fill the band. A hidden event is still the whole band in grey. The same reader is retrained on that picture. A second reader freezes ImageNet ResNet-18 through its first residual stage, stretches each cell to a square, and trains only the coupler.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | taller | goals | 730 | 740 | 189 | 249 | 383 |
| 20% | taller | assists | 707 | 678 | 100 | 158 | 221 |
| 20% | imagenet | goals | 730 | 661 | 182 | 241 | 364 |
| 20% | imagenet | assists | 707 | 601 | 89 | 147 | 195 |
| 80% | taller | goals | 199 | 200 | 55 | 71 | 99 |
| 80% | imagenet | goals | 199 | 176 | 49 | 66 | 92 |

At 20% revealed the taller bands put 189 goal bars on the same appearance and 249 within one gameweek. The ImageNet reader puts 182 and 241. The gate puts 190 and 258. The gate stays the stripped card measured at 15:53.

The table below is an older per-player ranking. It is a record of past runs. It is not the score.

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
| 8 Nov 2026, 22:16 | squad vision, flat marks and match budget | squad card, all minutes | 22.1% | 37.6% | 75.0% |
| 9 Nov 2026, 07:54 | squad vision, ranked fixture budget | squad card, all minutes | 24.7% | 41.5% | 80.1% |

Best on the player card in that old ranking: minutes, 11.4%. Best on the squad card in that old ranking: squad vision with the marked chart, 13.1%. The current score is the card comparison at the top. The reader is still squad vision with squad coupling.

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

## Flat marks and a match budget

The 22:16 run on 8 Nov keeps the locked vision reader and changes the event paint. A goal or assist band is white, red, green, or blue for 0, 1, 2, or 3. Hidden events stay gray. The conceded-rate band is hatched after the cut. A second head predicts the fixture total, and that total is placed onto players with minutes in order of expected count. The cell loss stays.

On the all-minutes exam the top 10% is 22.1%. The locked default stays at 24.7%. Cell samples on gameweeks 20–38 draw 464.7 goal bars against 459 actual, cover 15.1%, and land 14.9% on a real week. Budget placement draws 533.6 bars, covers 25.4%, and lands 21.8% on a real week. A same-count shuffle of the real bars covers 20.3%.

Goal remainders from gameweek 1, budget placement: at 80% revealed, 199 actual and 223.4 drawn, 27.9% covered, 24.8% on a real week; at 20% revealed, 730 actual and 823.9 drawn, 25.3% covered, 22.4% on a real week. Cell samples at those cuts draw 197.6 and 730.5. A goalkeeper's hidden appearance is a goal bar at 1.0%. A forward is at 12.7% when 80% is revealed and at 12.2% when 20% is revealed.

## Ranked fixture budget

The 07:54 run on 9 Nov retrains the locked vision reader. Its top 10% reproduces at 24.7%, so the capture gate is unchanged. The new piece is the decoder and the score.

Each future fixture receives a number of bars equal to the sum of its bar probabilities. Rounding is done so the fixtures add up to the season total. Those bars go to the players in that fixture with the highest probabilities and with minutes.

A blank zero is left out of the score. Precision is the share of painted bars on a real event. Recall is the share of real events that were painted. The harmonic mean is zero when the card is left blank, and it falls when extra bars land on zeros.

On gameweeks 20–38 the ranked decoder draws 440 goal bars against 459. Precision is 27.0%, recall is 25.9%, and the harmonic mean is 26.5%. Cell samples on the same probabilities draw 464.9 bars, with precision 14.8%, recall 14.9%, and a harmonic mean of 14.9%. A same-count shuffle scores 19.6% on all three.

From gameweek 1, ranked goals: at 80% revealed, 185 drawn against 199, precision 29.2%, recall 27.1%, harmonic mean 28.1%. At 20% revealed, 676 drawn against 730, precision 27.4%, recall 25.3%, harmonic mean 26.3%. A goalkeeper is at 0.1% and a forward is at 11.3% when 80% is revealed.

## Temporal distance

The 12:42 run on 9 Nov retrains the locked reader and measures where the misses sit. The top 10% reproduces at 24.7%, so the capture gate is unchanged. Mean hidden appearances in that exam: 18.8.

Of a player's hidden goal weeks, 24.7% are inside the vision top 10%, 0.9% are the other fixture in that gameweek, 17.3% are one gameweek away, 13.0% are two away, and 44.0% are three or more away. Of the weeks outside the top 10%, 24.2% are within one gameweek and 41.5% are within two. A random top 10% leaves 21.5% of its misses within one gameweek. Minutes leaves 27.7%, from an exact rate of 21.6%.

Matching a player's expected goals to the actual goals costs 4.29 gameweeks on average. A flat spread costs 4.86 and a permutation of the predicted spikes costs 4.90. For 1.5% of players the move is one gameweek or less. The probability of a bar has the same mean transport.

The ranked fixture bars land on 17.9% of a player's goal weeks. Of the other weeks, 9.7% are within one gameweek, against 10.2% for the same count on random weeks. 71.0% are three or more gameweeks away. A player who receives no bar counts as a miss on every goal week in this average.

Assists, same player rule: top 10% exact 20.2%, and 24.3% of the misses are within one gameweek. Expected-assist transport is 4.67 gameweeks, against 5.16 uniform and 5.19 permuted.
