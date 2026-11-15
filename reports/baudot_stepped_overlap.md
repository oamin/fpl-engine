# Stepped and horizon row opacity

The loss on a dot is the squared gap, (opacity − truth)². A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04.

Both arms paint rows. The stepped arm paints one gameweek: past rows stay true, and events from that week onward are blank. The horizon arm paints every week beyond a cutoff in one shot, and scores every live remainder dot equally. There is no column isolation. Header and minutes stay on every row. Player seats are shuffled in training.

Combined gives the one-week gap and the remainder gap equal weight.

## Next week and further

Stepped: one gameweek, stepped. Horizon: the remainder beyond the cutoff, one shot.

| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|---:|
| stepped | 1 | 0.0168 | 0.116 | 0.025 | 0.041 |
| stepped | 2 | 0.0141 | 0.114 | 0.025 | 0.040 |
| stepped | 4 | 0.0140 | 0.101 | 0.026 | 0.041 |
| stepped | 8 | 0.0155 | 0.118 | 0.025 | 0.038 |
| horizon | 1 | 0.0169 | 0.099 | 0.020 | 0.013 |
| horizon | 2 | 0.0137 | 0.101 | 0.019 | 0.012 |
| horizon | 4 | 0.0137 | 0.087 | 0.020 | 0.012 |
| horizon | 8 | 0.0155 | 0.098 | 0.019 | 0.012 |
| combined | 1 | 0.0169 | 0.094 | 0.020 | 0.018 |
| combined | 2 | 0.0137 | 0.096 | 0.020 | 0.017 |
| combined | 4 | 0.0137 | 0.083 | 0.021 | 0.018 |
| combined | 8 | 0.0155 | 0.098 | 0.020 | 0.017 |

## Remainder window

A week eight ahead is read from the one-shot remainder. Events between the cutoff and that week stay blank. Header and minutes of those weeks stay.

| arm | remainder gap | opacity on solid | opacity on blank | GKP goal opacity |
|---|---:|---:|---:|---:|
| stepped | 0.0145 | 0.152 | 0.025 | 0.038 |
| horizon | 0.0143 | 0.130 | 0.019 | 0.012 |
| combined | 0.0144 | 0.122 | 0.020 | 0.018 |

## Event marker, one week ahead

The first goal dot and the first assist dot.

| arm | revealed | channel | squared gap | opacity on solid | opacity on blank | GKP |
|---|---|---|---:|---:|---:|---:|
| stepped | 20% | goals | 0.0494 | 0.160 | 0.066 | 0.047 |
| stepped | 40% | goals | 0.0412 | 0.223 | 0.062 | 0.040 |
| stepped | 60% | goals | 0.0460 | 0.208 | 0.061 | 0.042 |
| stepped | 80% | goals | 0.0427 | 0.214 | 0.063 | 0.036 |
| stepped | 20% | assists | 0.0517 | 0.118 | 0.060 | 0.053 |
| stepped | 40% | assists | 0.0497 | 0.104 | 0.058 | 0.048 |
| stepped | 60% | assists | 0.0439 | 0.112 | 0.058 | 0.050 |
| stepped | 80% | assists | 0.0416 | 0.134 | 0.056 | 0.045 |
| horizon | 20% | goals | 0.0501 | 0.125 | 0.045 | 0.013 |
| horizon | 40% | goals | 0.0415 | 0.199 | 0.045 | 0.013 |
| horizon | 60% | goals | 0.0475 | 0.154 | 0.044 | 0.012 |
| horizon | 80% | goals | 0.0420 | 0.202 | 0.050 | 0.013 |
| horizon | 20% | assists | 0.0523 | 0.097 | 0.045 | 0.022 |
| horizon | 40% | assists | 0.0492 | 0.098 | 0.045 | 0.021 |
| horizon | 60% | assists | 0.0430 | 0.108 | 0.043 | 0.021 |
| horizon | 80% | assists | 0.0415 | 0.134 | 0.048 | 0.021 |
| combined | 20% | goals | 0.0497 | 0.112 | 0.043 | 0.018 |
| combined | 40% | goals | 0.0419 | 0.176 | 0.044 | 0.018 |
| combined | 60% | goals | 0.0462 | 0.164 | 0.044 | 0.018 |
| combined | 80% | goals | 0.0425 | 0.178 | 0.048 | 0.017 |
| combined | 20% | assists | 0.0521 | 0.096 | 0.046 | 0.028 |
| combined | 40% | assists | 0.0497 | 0.092 | 0.048 | 0.028 |
| combined | 60% | assists | 0.0437 | 0.099 | 0.047 | 0.028 |
| combined | 80% | assists | 0.0415 | 0.126 | 0.048 | 0.028 |

## Call

Next-week squared gap: stepped **0.0168**, horizon **0.0169**, combined **0.0169**. Remainder gap, every live dot: stepped **0.0145**, horizon **0.0143**, combined **0.0144**. Goalkeeper goal opacity on the next week: stepped **0.041**, horizon **0.013**, combined **0.018**. The stepped card steps one gameweek at a time. The horizon card is one shot of the weeks beyond the cutoff.

Example cards: `reports/baudot_step_stepped_arsenal_2025-26.png`, `reports/baudot_step_horizon_arsenal_2025-26.png`, `reports/baudot_step_product_arsenal_2025-26.png`, `reports/baudot_step_product_stretched_arsenal_2025-26.png`.
