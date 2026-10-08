# Gameweek 6 anytime goalscorer trial

Historical `score_xp` is unchanged. This note replaces Gameweek 6 goals for players with a US anytime price. The price is the mean of 1/decimal, clipped, then converted with -ln(1-p), multiplied by minutes/90, and reduced when the priced players on a club would exceed the match line left after unpriced shares. A missing price keeps share times team λ. Gameweek 7 is the match line only. Gameweek 8 onward stays unknown: the Odds API sports list has no Premier League winner, top-four, or relegation market, and a two-match attack rate is not used as a ranking.

Requests, region `us`, market `player_goal_scorer_anytime`:

- Arsenal v Leeds United: HTTP 200, last 1, remaining 490
- Aston Villa v Brentford: HTTP 200, last 1, remaining 489
- Chelsea v Bournemouth: HTTP 200, last 1, remaining 488
- Sunderland v Brighton and Hove Albion: HTTP 200, last 1, remaining 487
- Ipswich Town v Fulham: HTTP 200, last 1, remaining 486
- Manchester United v Tottenham Hotspur: HTTP 200, last 1, remaining 485
- Crystal Palace v Nottingham Forest: HTTP 200, last 1, remaining 484
- Hull City v Everton: HTTP 200, last 1, remaining 483
- Liverpool v Manchester City: HTTP 200, last 1, remaining 482
- Coventry City v Newcastle United: HTTP 200, last 1, remaining 481

Quoted names 474. Matched 373. Unmatched 101.

## Four players

| Player | Tag | Minutes | Old goals | Book goals | Old GW6 | Book GW6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Calafiori | minutes file | 90 | 0.16 | 0.12 | 5.16 | 4.85 |
| Haaland | minutes file | 90 | 0.58 | 0.58 | 4.80 | 4.80 |
| Palmer | doubtful | 49 | 0.34 | 0.25 | 4.65 | 4.11 |
| Mbeumo | last observed | 90 | 0.57 | 0.40 | 6.24 | 5.25 |

## Plans on the match line

| Plan | GW6 XI | GW6 bench | GW7 XI | GW7 bench |
| --- | ---: | ---: | ---: | ---: |
| Hold | 56.09 | 4.85 | 58.90 | 3.63 |
| Free transfer | 58.10 | 6.16 | 61.26 | 4.95 |
| Wildcard | 69.16 | 17.33 | 63.41 | 14.91 |

Free transfer: sells van Ewijk (DEF, COV, 4.0); buys Hall (DEF, NEW, 5.3).

## Plans with Gameweek 6 book goals

| Plan | GW6 XI | GW6 bench | GW7 XI | GW7 bench |
| --- | ---: | ---: | ---: | ---: |
| Hold | 54.18 | 4.68 | 58.90 | 3.63 |
| Free transfer | 56.62 | 5.43 | 61.26 | 4.95 |
| Wildcard | 61.91 | 15.59 | 61.26 | 12.51 |

Free transfer (1 move): sells van Ewijk (DEF, COV, 4.0); buys Hall (DEF, NEW, 5.3).

Wildcard sells Forster (GKP, BOU, 4.0), Lammens (GKP, MUN, 4.9), Calafiori (DEF, ARS, 5.7), Davis (DEF, IPS, 4.0), Guéhi (DEF, MCI, 6.0), Shaw (DEF, MUN, 4.3), van Ewijk (DEF, COV, 4.0), Barnes (MID, NEW, 6.0), Cherki (MID, MCI, 7.6), Calvert-Lewin (FWD, LEE, 6.0), Haaland (FWD, MCI, 15.5), Scarlett (FWD, TOT, 4.5). Buys Martinez (GKP, CHE, 5.0), Verbruggen (GKP, BHA, 4.5), Gabriel (DEF, ARS, 8.0), Hall (DEF, NEW, 5.3), Khalaili (DEF, CRY, 5.0), Murillo (DEF, NFO, 5.5), Thiaw (DEF, NEW, 5.0), Mbeumo (MID, MUN, 7.9), Saka (MID, ARS, 9.6), Barry (FWD, EVE, 5.7), N.Jackson (FWD, AVL, 6.5), Welbeck (FWD, CHE, 5.9).

Two-week XI sums, match line then book goals: hold 114.99 to 113.08, free transfer 119.36 to 117.88, wildcard 132.58 to 123.17.

This is a one-week overlay. It does not count toward a chip rule.
