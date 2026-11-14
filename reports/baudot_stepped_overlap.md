# Stepped per-dot gap

The loss on a dot is the squared gap, (opacity − truth)². A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04.

The row step scores the next gameweek, every player who plays it. The column step scores each player's future weeks, and each player counts once. Week t is scored with the paint at week t. That paint sees the week's minutes and opponent, so the gray moves with the match. A single gray repeated down the player ignores the match. That version is withdrawn.

Combined gives the next-week gap and the column gap equal weight.

## Next week and further

This is the row question. One week, read with the weeks between left blank.

| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|---:|
| row | 1 | 0.0168 | 0.101 | 0.020 | 0.021 |
| row | 2 | 0.0137 | 0.106 | 0.020 | 0.020 |
| row | 4 | 0.0137 | 0.088 | 0.020 | 0.021 |
| row | 8 | 0.0155 | 0.100 | 0.019 | 0.021 |
| column | 1 | 0.0168 | 0.100 | 0.020 | 0.015 |
| column | 2 | 0.0136 | 0.103 | 0.019 | 0.014 |
| column | 4 | 0.0137 | 0.085 | 0.020 | 0.015 |
| column | 8 | 0.0154 | 0.100 | 0.019 | 0.015 |
| combined | 1 | 0.0169 | 0.104 | 0.021 | 0.017 |
| combined | 2 | 0.0137 | 0.108 | 0.021 | 0.016 |
| combined | 4 | 0.0138 | 0.089 | 0.021 | 0.016 |
| combined | 8 | 0.0155 | 0.104 | 0.020 | 0.017 |

## Each player's future column

Each future week is scored on its own. Minutes and the opponent stay on the card. Goals and assists between the cut and that week stay blank.

| arm | column gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|
| row | 0.0143 | 0.105 | 0.020 | 0.020 |
| column | 0.0142 | 0.105 | 0.019 | 0.014 |
| combined | 0.0143 | 0.109 | 0.021 | 0.016 |

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
| column | 20% | goals | 0.0501 | 0.125 | 0.047 | 0.016 |
| column | 40% | goals | 0.0415 | 0.201 | 0.047 | 0.015 |
| column | 60% | goals | 0.0474 | 0.158 | 0.045 | 0.015 |
| column | 80% | goals | 0.0416 | 0.201 | 0.050 | 0.014 |
| column | 20% | assists | 0.0527 | 0.090 | 0.045 | 0.022 |
| column | 40% | assists | 0.0491 | 0.101 | 0.047 | 0.022 |
| column | 60% | assists | 0.0428 | 0.113 | 0.044 | 0.021 |
| column | 80% | assists | 0.0416 | 0.132 | 0.049 | 0.022 |
| combined | 20% | goals | 0.0495 | 0.132 | 0.046 | 0.017 |
| combined | 40% | goals | 0.0414 | 0.203 | 0.046 | 0.017 |
| combined | 60% | goals | 0.0463 | 0.181 | 0.046 | 0.017 |
| combined | 80% | goals | 0.0430 | 0.196 | 0.050 | 0.016 |
| combined | 20% | assists | 0.0524 | 0.101 | 0.048 | 0.029 |
| combined | 40% | assists | 0.0496 | 0.099 | 0.048 | 0.028 |
| combined | 60% | assists | 0.0433 | 0.112 | 0.048 | 0.029 |
| combined | 80% | assists | 0.0410 | 0.139 | 0.049 | 0.028 |

## Call

Next-week squared gap: row **0.0168**, column **0.0168**, combined **0.0169**. Column gap, each player once down the future: row **0.0143**, column **0.0142**, combined **0.0143**. Goalkeeper goal opacity on the next week: row **0.021**, column **0.015**, combined **0.017**. On the column, each week keeps its own minutes and opponent.

Example cards: `reports/baudot_step_row_arsenal_2025-26.png`, `reports/baudot_step_column_arsenal_2025-26.png`, `reports/baudot_step_row_liverpool_2025-26.png`, `reports/baudot_step_column_liverpool_2025-26.png`. On the pictured attackers the goal dot moves by about 0.08 from week to week. The Arsenal column forward runs from pixel 133 to 216. The goalkeeper stays at 215–221.
