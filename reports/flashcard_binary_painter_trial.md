# Binary card painters

The training set is the binary card. Odds, venue, minutes, goals,
and assists are black and white. A hidden event square is a checker.
The picture painter writes a color on every hidden square in one
pass. The restored painter starts those squares as noise and rebuilds
the picture in 32 steps. A bar is a square closer to black than to
white or to the checker. The other squad is not an input.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

The color score is the same painter on the color card. The gate is
the 21:47 UK full painter. This run does not replace it.

## Picture painter

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 0 | 0 | 0 | 0 |
| 20% | assists | 707 | 0 | 0 | 0 | 0 |
| 40% | goals | 544 | 0 | 0 | 0 | 0 |
| 40% | assists | 531 | 0 | 0 | 0 | 0 |
| 60% | goals | 360 | 0 | 0 | 0 | 0 |
| 60% | assists | 351 | 0 | 0 | 0 | 0 |
| 80% | goals | 199 | 0 | 0 | 0 | 0 |
| 80% | assists | 192 | 0 | 0 | 0 | 0 |

| revealed | channel | actual bars | binary bars | binary hits | binary score | gate score | color picture score |
|---|---|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 0 | 0 | 0.000 | 0.248 | 0.000 |
| 20% | assists | 707 | 0 | 0 | 0.000 | 0.140 | 0.000 |
| 40% | goals | 544 | 0 | 0 | 0.000 | 0.240 | 0.000 |
| 40% | assists | 531 | 0 | 0 | 0.000 | 0.145 | 0.000 |
| 60% | goals | 360 | 0 | 0 | 0.000 | 0.237 | 0.000 |
| 60% | assists | 351 | 0 | 0 | 0.000 | 0.156 | 0.000 |
| 80% | goals | 199 | 0 | 0 | 0.000 | 0.266 | 0.000 |
| 80% | assists | 192 | 0 | 0 | 0.000 | 0.184 | 0.000 |

## Restored painter

| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 635 | 107 | 234 | 241 |
| 20% | assists | 707 | 569 | 71 | 152 | 160 |
| 40% | goals | 544 | 502 | 99 | 192 | 201 |
| 40% | assists | 531 | 438 | 57 | 121 | 123 |
| 60% | goals | 360 | 324 | 70 | 125 | 135 |
| 60% | assists | 351 | 284 | 36 | 80 | 79 |
| 80% | goals | 199 | 191 | 49 | 78 | 81 |
| 80% | assists | 192 | 162 | 19 | 39 | 36 |

| revealed | channel | actual bars | binary bars | binary hits | binary score | gate score | color restored score |
|---|---|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 635 | 107 | 0.136 | 0.248 | 0.168 |
| 20% | assists | 707 | 569 | 71 | 0.090 | 0.140 | 0.035 |
| 40% | goals | 544 | 502 | 99 | 0.175 | 0.240 | 0.177 |
| 40% | assists | 531 | 438 | 57 | 0.097 | 0.145 | 0.039 |
| 60% | goals | 360 | 324 | 70 | 0.184 | 0.237 | 0.188 |
| 60% | assists | 351 | 284 | 36 | 0.092 | 0.156 | 0.035 |
| 80% | goals | 199 | 191 | 49 | 0.241 | 0.266 | 0.221 |
| 80% | assists | 192 | 162 | 19 | 0.091 | 0.184 | 0.035 |

Squares are listed from the bottom. Zero minutes is an appearance
with no minutes. Played is an appearance with minutes.

### Picture painter

| source | minutes | 1st square | 2nd square | 3rd square | any square |
|---|---|---:|---:|---:|---:|
| 20% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual goals | played | 0.081 | 0.008 | 0.001 | 0.081 |
| 20% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual assists | played | 0.078 | 0.005 | 0.000 | 0.078 |
| 20% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted goals | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted assists | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual goals | played | 0.079 | 0.007 | 0.001 | 0.079 |
| 40% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual assists | played | 0.077 | 0.004 | 0.000 | 0.077 |
| 40% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted goals | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted assists | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual goals | played | 0.080 | 0.007 | 0.001 | 0.080 |
| 60% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual assists | played | 0.078 | 0.004 | 0.000 | 0.078 |
| 60% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted goals | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted assists | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual goals | played | 0.082 | 0.008 | 0.000 | 0.082 |
| 80% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual assists | played | 0.079 | 0.005 | 0.000 | 0.079 |
| 80% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted goals | played | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted assists | played | 0.000 | 0.000 | 0.000 | 0.000 |

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 0 | 0 | 0 | 0 |
| 40% | 0 | 0 | 0 | 0 |
| 60% | 0 | 0 | 0 | 0 |
| 80% | 0 | 0 | 0 | 0 |

### Restored painter

| source | minutes | 1st square | 2nd square | 3rd square | any square |
|---|---|---:|---:|---:|---:|
| 20% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual goals | played | 0.081 | 0.008 | 0.001 | 0.081 |
| 20% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% actual assists | played | 0.078 | 0.005 | 0.000 | 0.078 |
| 20% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted goals | played | 0.070 | 0.006 | 0.000 | 0.070 |
| 20% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 20% painted assists | played | 0.063 | 0.001 | 0.000 | 0.063 |
| 40% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual goals | played | 0.079 | 0.007 | 0.001 | 0.079 |
| 40% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% actual assists | played | 0.077 | 0.004 | 0.000 | 0.077 |
| 40% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted goals | played | 0.073 | 0.006 | 0.000 | 0.073 |
| 40% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 40% painted assists | played | 0.063 | 0.001 | 0.000 | 0.063 |
| 60% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual goals | played | 0.080 | 0.007 | 0.001 | 0.080 |
| 60% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% actual assists | played | 0.078 | 0.004 | 0.000 | 0.078 |
| 60% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted goals | played | 0.072 | 0.007 | 0.000 | 0.072 |
| 60% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 60% painted assists | played | 0.063 | 0.002 | 0.000 | 0.063 |
| 80% actual goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual goals | played | 0.082 | 0.008 | 0.000 | 0.082 |
| 80% actual assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% actual assists | played | 0.079 | 0.005 | 0.000 | 0.079 |
| 80% painted goals | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted goals | played | 0.079 | 0.008 | 0.000 | 0.079 |
| 80% painted assists | zero | 0.000 | 0.000 | 0.000 | 0.000 |
| 80% painted assists | played | 0.067 | 0.002 | 0.000 | 0.067 |

| revealed | goal stacks | skipping | assist stacks | skipping |
|---|---:|---:|---:|---:|
| 20% | 635 | 0 | 569 | 0 |
| 40% | 502 | 0 | 438 | 0 |
| 60% | 324 | 0 | 284 | 0 |
| 80% | 191 | 0 | 162 | 0 |
