# Stepped per-dot gap

The loss on a dot is the squared gap, (opacity − truth)². A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04.

The row step looks at the shared teamsheet, plus this player's position and minutes. It does not see the opponent. His own name is hidden, and his past games are hidden. Two players of the same position with the same minutes get the same gray. The column step looks at that player's earlier games, then this week's minutes and opponent. The name is hidden, and the teammates are hidden. Both score the squared gap on the week they paint.

Combined paints the mean of the two probabilities and scores the squared gap of that mean.

## Next week and further

Row: the other names on the teamsheet. Column: the player's earlier games.

| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|---:|
| row | 1 | 0.0171 | 0.097 | 0.018 | 0.007 |
| row | 2 | 0.0138 | 0.103 | 0.018 | 0.007 |
| row | 4 | 0.0138 | 0.090 | 0.019 | 0.007 |
| row | 8 | 0.0156 | 0.100 | 0.018 | 0.007 |
| column | 1 | 0.0174 | 0.077 | 0.018 | 0.049 |
| column | 2 | 0.0142 | 0.075 | 0.018 | 0.049 |
| column | 4 | 0.0141 | 0.062 | 0.018 | 0.048 |
| column | 8 | 0.0160 | 0.075 | 0.017 | 0.051 |
| combined | 1 | 0.0170 | 0.098 | 0.018 | 0.016 |
| combined | 2 | 0.0137 | 0.102 | 0.018 | 0.015 |
| combined | 4 | 0.0138 | 0.087 | 0.018 | 0.015 |
| combined | 8 | 0.0155 | 0.097 | 0.018 | 0.015 |

## Each player's future column

For a week eight ahead, the column still uses only the games played before the cut.

| arm | column gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|
| row | 0.0145 | 0.109 | 0.018 | 0.006 |
| column | 0.0148 | 0.089 | 0.017 | 0.049 |
| combined | 0.0143 | 0.107 | 0.018 | 0.015 |

## Event marker, one week ahead

The first goal dot and the first assist dot.

| arm | revealed | channel | squared gap | opacity on solid | opacity on blank | GKP |
|---|---|---|---:|---:|---:|---:|
| row | 20% | goals | 0.0497 | 0.144 | 0.051 | 0.007 |
| row | 40% | goals | 0.0425 | 0.191 | 0.051 | 0.007 |
| row | 60% | goals | 0.0492 | 0.161 | 0.048 | 0.006 |
| row | 80% | goals | 0.0442 | 0.169 | 0.049 | 0.007 |
| row | 20% | assists | 0.0525 | 0.102 | 0.048 | 0.009 |
| row | 40% | assists | 0.0503 | 0.094 | 0.049 | 0.009 |
| row | 60% | assists | 0.0444 | 0.101 | 0.046 | 0.008 |
| row | 80% | assists | 0.0418 | 0.114 | 0.046 | 0.009 |
| column | 20% | goals | 0.0510 | 0.106 | 0.048 | 0.048 |
| column | 40% | goals | 0.0438 | 0.155 | 0.052 | 0.049 |
| column | 60% | goals | 0.0512 | 0.108 | 0.048 | 0.049 |
| column | 80% | goals | 0.0469 | 0.109 | 0.050 | 0.051 |
| column | 20% | assists | 0.0528 | 0.086 | 0.048 | 0.049 |
| column | 40% | assists | 0.0484 | 0.102 | 0.049 | 0.050 |
| column | 60% | assists | 0.0432 | 0.096 | 0.047 | 0.050 |
| column | 80% | assists | 0.0437 | 0.089 | 0.048 | 0.052 |
| combined | 20% | goals | 0.0491 | 0.144 | 0.051 | 0.016 |
| combined | 40% | goals | 0.0415 | 0.202 | 0.050 | 0.015 |
| combined | 60% | goals | 0.0484 | 0.163 | 0.049 | 0.017 |
| combined | 80% | goals | 0.0436 | 0.168 | 0.049 | 0.016 |
| combined | 20% | assists | 0.0524 | 0.103 | 0.049 | 0.015 |
| combined | 40% | assists | 0.0498 | 0.097 | 0.049 | 0.014 |
| combined | 60% | assists | 0.0438 | 0.105 | 0.047 | 0.015 |
| combined | 80% | assists | 0.0418 | 0.112 | 0.045 | 0.015 |

## Call

Next-week squared gap: row **0.0171**, column **0.0174**, combined **0.0170**. Column gap, each player once down the future: row **0.0145**, column **0.0148**, combined **0.0143**. Goalkeeper goal opacity on the next week: row **0.007**, column **0.049**, combined **0.016**. The row card is the teamsheet: same position and the same minutes share a gray, and the opponent is hidden. The column card is that player's earlier games, then this week's minutes and opponent. On the Arsenal card, where both are painting a player who is on the pitch, the goal grays correlate at 0.42. The row paints the striker dark because of his position. The column paints him from his earlier games, so that column is lighter.

Example cards: `reports/baudot_step_row_arsenal_2025-26.png`, `reports/baudot_step_column_arsenal_2025-26.png`, `reports/baudot_step_row_liverpool_2025-26.png`, `reports/baudot_step_column_liverpool_2025-26.png`.
