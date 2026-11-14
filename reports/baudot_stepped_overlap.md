# Stepped per-dot gap

The loss on a dot is the squared gap, (opacity − truth)². A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04.

The row step scores the next gameweek. Each player can take a different gray that week, and the gray can change the week after. The column step gives a player one gray for the whole future column and scores that gray against every week in it. A goal and a blank pull the same dot. Giving every week its own gray is the row again, which is why the two cards matched.

Combined gives the next-week gap and the column gap equal weight.

## Next week and further

This is the row question. One week, read with the weeks between left blank.

| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|---:|
| row | 1 | 0.0168 | 0.101 | 0.020 | 0.021 |
| row | 2 | 0.0137 | 0.106 | 0.020 | 0.020 |
| row | 4 | 0.0137 | 0.088 | 0.020 | 0.021 |
| row | 8 | 0.0155 | 0.100 | 0.019 | 0.021 |
| column | 1 | 0.0168 | 0.104 | 0.020 | 0.015 |
| column | 2 | 0.0137 | 0.111 | 0.020 | 0.014 |
| column | 4 | 0.0138 | 0.090 | 0.020 | 0.014 |
| column | 8 | 0.0154 | 0.104 | 0.019 | 0.014 |
| combined | 1 | 0.0169 | 0.105 | 0.021 | 0.018 |
| combined | 2 | 0.0138 | 0.108 | 0.021 | 0.018 |
| combined | 4 | 0.0138 | 0.090 | 0.021 | 0.018 |
| combined | 8 | 0.0155 | 0.104 | 0.020 | 0.018 |

## Each player's future column

One gray per player, compared with every later week. The weeks between the cut and that week stay blank.

| arm | column gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|
| row | 0.0147 | 0.078 | 0.020 | 0.020 |
| column | 0.0146 | 0.080 | 0.020 | 0.014 |
| combined | 0.0147 | 0.080 | 0.021 | 0.017 |

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
| column | 20% | goals | 0.0502 | 0.131 | 0.048 | 0.015 |
| column | 40% | goals | 0.0409 | 0.204 | 0.046 | 0.014 |
| column | 60% | goals | 0.0467 | 0.179 | 0.047 | 0.015 |
| column | 80% | goals | 0.0422 | 0.197 | 0.048 | 0.014 |
| column | 20% | assists | 0.0532 | 0.093 | 0.049 | 0.024 |
| column | 40% | assists | 0.0491 | 0.105 | 0.049 | 0.023 |
| column | 60% | assists | 0.0432 | 0.120 | 0.049 | 0.023 |
| column | 80% | assists | 0.0413 | 0.134 | 0.048 | 0.022 |
| combined | 20% | goals | 0.0494 | 0.137 | 0.047 | 0.018 |
| combined | 40% | goals | 0.0415 | 0.200 | 0.047 | 0.018 |
| combined | 60% | goals | 0.0462 | 0.185 | 0.047 | 0.018 |
| combined | 80% | goals | 0.0430 | 0.194 | 0.050 | 0.017 |
| combined | 20% | assists | 0.0525 | 0.103 | 0.050 | 0.031 |
| combined | 40% | assists | 0.0498 | 0.098 | 0.051 | 0.030 |
| combined | 60% | assists | 0.0435 | 0.112 | 0.050 | 0.031 |
| combined | 80% | assists | 0.0409 | 0.140 | 0.050 | 0.029 |

## Call

Next-week squared gap: row **0.0168**, column **0.0168**, combined **0.0169**. Column gap, each player once down the future: row **0.0147**, column **0.0146**, combined **0.0147**. Goalkeeper goal opacity on the next week: row **0.021**, column **0.015**, combined **0.018**. The row card changes shade from week to week. The column card repeats one shade down the player. The average gap does not pick a winner: a week of its own and one gray for the column miss by about the same amount.

Example cards: `reports/baudot_step_row_arsenal_2025-26.png`, `reports/baudot_step_column_arsenal_2025-26.png`, `reports/baudot_step_row_liverpool_2025-26.png`, `reports/baudot_step_column_liverpool_2025-26.png`.
