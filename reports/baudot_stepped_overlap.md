# Stepped per-dot gap

The set-overlap score is withdrawn. It can improve by painting a likely scorer solid on the weeks he does not score, which filled the attacker columns and, on the column arm, the goalkeeper squares.

The loss is now the squared gap on each dot, (opacity − truth)². A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04. The minimum is the probability of the dot. Row averages the dots of the week. Column averages each player first, so a goalkeeper's squares weigh the same as a forward's. Combined gives the two averages equal weight.

Opacity on a solid dot should sit near the chance of that event. Opacity on a blank dot, and on a goalkeeper's goal dot, should sit near zero.

## Next week and further

| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|---:|
| row | 1 | 0.0168 | 0.101 | 0.020 | 0.021 |
| row | 2 | 0.0137 | 0.106 | 0.020 | 0.020 |
| row | 4 | 0.0137 | 0.088 | 0.020 | 0.021 |
| row | 8 | 0.0155 | 0.100 | 0.019 | 0.021 |
| column | 1 | 0.0168 | 0.101 | 0.020 | 0.021 |
| column | 2 | 0.0137 | 0.106 | 0.020 | 0.020 |
| column | 4 | 0.0137 | 0.088 | 0.020 | 0.021 |
| column | 8 | 0.0155 | 0.100 | 0.019 | 0.021 |
| combined | 1 | 0.0168 | 0.101 | 0.020 | 0.021 |
| combined | 2 | 0.0137 | 0.106 | 0.020 | 0.020 |
| combined | 4 | 0.0137 | 0.088 | 0.020 | 0.021 |
| combined | 8 | 0.0155 | 0.100 | 0.019 | 0.021 |

## Event marker, one week ahead

The first goal dot and the first assist dot.

| arm | revealed | channel | squared gap | opacity on solid | opacity on blank | GKP |
|---|---|---|---:|---:|---:|---:|
| row | 20% | goals | 0.0498 | 0.124 | 0.047 | 0.022 |
| row | 40% | goals | 0.0411 | 0.203 | 0.047 | 0.020 |
| row | 60% | goals | 0.0463 | 0.180 | 0.048 | 0.021 |
| row | 80% | goals | 0.0429 | 0.195 | 0.050 | 0.019 |
| row | 20% | assists | 0.0521 | 0.099 | 0.049 | 0.035 |
| row | 40% | assists | 0.0495 | 0.094 | 0.049 | 0.033 |
| row | 60% | assists | 0.0434 | 0.106 | 0.050 | 0.034 |
| row | 80% | assists | 0.0411 | 0.131 | 0.050 | 0.032 |
| column | 20% | goals | 0.0498 | 0.124 | 0.047 | 0.022 |
| column | 40% | goals | 0.0411 | 0.203 | 0.047 | 0.020 |
| column | 60% | goals | 0.0463 | 0.180 | 0.048 | 0.021 |
| column | 80% | goals | 0.0429 | 0.195 | 0.050 | 0.019 |
| column | 20% | assists | 0.0521 | 0.099 | 0.049 | 0.035 |
| column | 40% | assists | 0.0495 | 0.094 | 0.049 | 0.033 |
| column | 60% | assists | 0.0434 | 0.106 | 0.050 | 0.034 |
| column | 80% | assists | 0.0411 | 0.131 | 0.050 | 0.032 |
| combined | 20% | goals | 0.0498 | 0.124 | 0.047 | 0.022 |
| combined | 40% | goals | 0.0411 | 0.203 | 0.047 | 0.020 |
| combined | 60% | goals | 0.0463 | 0.180 | 0.048 | 0.021 |
| combined | 80% | goals | 0.0429 | 0.195 | 0.050 | 0.019 |
| combined | 20% | assists | 0.0521 | 0.099 | 0.049 | 0.035 |
| combined | 40% | assists | 0.0495 | 0.094 | 0.049 | 0.033 |
| combined | 60% | assists | 0.0434 | 0.106 | 0.050 | 0.034 |
| combined | 80% | assists | 0.0411 | 0.131 | 0.050 | 0.032 |

## Call

Next-week squared gap: row **0.0168**, column **0.0168**, combined **0.0168**. Blank-dot opacity **0.020**. Opacity on a solid dot **0.101**. Goalkeeper goal opacity **0.021**. The three arms match. Every player has six event dots, so the mean over dots and the mean over players are the same number, and the combined arm is that number again. The column split only changed the old overlap ratio. A true goal marker at the 20% cut is **0.124** opaque, and a blank goal marker is **0.047**.

Example cards, each remainder week painted as its own step: `reports/baudot_step_row_arsenal_2025-26.png`, `reports/baudot_step_column_arsenal_2025-26.png`, `reports/baudot_step_row_liverpool_2025-26.png`, `reports/baudot_step_column_liverpool_2025-26.png`. On the pictured Arsenal regulars the solid dots are 0.18 opaque, the blank dots 0.11, and the goalkeeper goal dots 0.04. The row card and the column card match.
