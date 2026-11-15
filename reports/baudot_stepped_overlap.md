# Stepped per-dot gap

The loss on a dot is the squared gap, (opacity − truth)². A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04.

The row step paints one gameweek. Past rows stay true, and events from that week onward are blank. The column step paints every week beyond a cutoff in one shot, and the loss is each player's column. Header and minutes stay on every row. Both score the squared gap between the gray and the true dots.

Combined gives the one-week gap and the remainder-column gap equal weight.

## Next week and further

Row: one gameweek, stepped. Column: the remainder beyond the cutoff, one shot.

| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|---:|
| row | 1 | 0.0168 | 0.101 | 0.020 | 0.021 |
| row | 2 | 0.0137 | 0.107 | 0.020 | 0.020 |
| row | 4 | 0.0137 | 0.090 | 0.020 | 0.020 |
| row | 8 | 0.0155 | 0.106 | 0.020 | 0.020 |
| column | 1 | 0.0168 | 0.100 | 0.020 | 0.015 |
| column | 2 | 0.0136 | 0.103 | 0.019 | 0.014 |
| column | 4 | 0.0137 | 0.085 | 0.020 | 0.015 |
| column | 8 | 0.0154 | 0.100 | 0.019 | 0.015 |
| combined | 1 | 0.0169 | 0.090 | 0.019 | 0.017 |
| combined | 2 | 0.0136 | 0.096 | 0.018 | 0.017 |
| combined | 4 | 0.0136 | 0.080 | 0.019 | 0.016 |
| combined | 8 | 0.0154 | 0.095 | 0.019 | 0.017 |

## Each player's future column

A week eight ahead is read from the one-shot remainder. Events between the cutoff and that week stay blank. Header and minutes of those weeks stay.

| arm | column gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|
| row | 0.0143 | 0.109 | 0.020 | 0.019 |
| column | 0.0142 | 0.105 | 0.019 | 0.014 |
| combined | 0.0142 | 0.096 | 0.018 | 0.017 |

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
| combined | 20% | goals | 0.0506 | 0.106 | 0.041 | 0.018 |
| combined | 40% | goals | 0.0418 | 0.180 | 0.042 | 0.017 |
| combined | 60% | goals | 0.0475 | 0.146 | 0.042 | 0.017 |
| combined | 80% | goals | 0.0418 | 0.179 | 0.045 | 0.016 |
| combined | 20% | assists | 0.0524 | 0.089 | 0.046 | 0.031 |
| combined | 40% | assists | 0.0493 | 0.090 | 0.047 | 0.030 |
| combined | 60% | assists | 0.0430 | 0.104 | 0.046 | 0.030 |
| combined | 80% | assists | 0.0413 | 0.122 | 0.048 | 0.030 |

## Call

Next-week squared gap: row **0.0168**, column **0.0168**, combined **0.0169**. Column gap, each player once down the future: row **0.0143**, column **0.0142**, combined **0.0142**. Goalkeeper goal opacity on the next week: row **0.021**, column **0.015**, combined **0.017**. The row card steps one gameweek at a time. The column card is one shot of the weeks beyond the cutoff.

Example cards: `reports/baudot_step_row_arsenal_2025-26.png`, `reports/baudot_step_column_arsenal_2025-26.png`, `reports/baudot_step_row_liverpool_2025-26.png`, `reports/baudot_step_column_liverpool_2025-26.png`.
