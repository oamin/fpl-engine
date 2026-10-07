# Score calibration, 2025/26

Published `compute_xp`, Gameweeks 5–38, players with at least 3 prior appearances and a minutes prior of at least 45. No coefficient was changed.

The official awards reconstructed from the sheet (appearance, goals, assists, clean sheets, defensive contributions, saves, bonus, goals conceded, cards) differ from `total_points` by **0.003** points per row on average. Own goals, penalty saves, and missed penalties are not on this reconstruction.

A goalkeeper goal is worth 6 in the official total and 10 inside `xp_goals`. That scale was left as it is so earlier xP totals stay reproducible.

## Score against actual points

| position | n | bias | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 7569 | +0.128 | 2.289 | 2.995 | 0.235 | 0.229 |
| GKP | 639 | +0.158 | 2.166 | 2.668 | 0.168 | 0.165 |
| DEF | 2899 | +0.298 | 2.433 | 3.057 | 0.236 | 0.229 |
| MID | 3265 | +0.056 | 2.150 | 2.922 | 0.248 | 0.265 |
| FWD | 766 | -0.229 | 2.437 | 3.308 | 0.251 | 0.222 |

Bias is predicted score minus actual points. A positive bias is a score that sits above the points.

## Components

Bias is the predicted component minus the official points from that event. `delta Spearman` is how much rank agreement with total points falls when the component is removed from the score. Keep means the mean error is inside 0.10 and the rank change is under 0.01.

| component | n | bias | MAE | RMSE | Spearman vs its own points | delta Spearman | keep |
|---|---:|---:|---:|---:|---:|---:|---|
| appear | 7569 | +0.160 | 0.197 | 0.417 | 0.278 | +0.006 | no |
| goals | 7569 | -0.022 | 0.754 | 1.543 | 0.241 | +0.046 | no |
| assists | 7569 | -0.065 | 0.426 | 0.900 | 0.160 | +0.012 | no |
| clean_sheets | 7569 | +0.226 | 0.876 | 1.162 | 0.222 | +0.023 | no |
| defcon | 7569 | -0.025 | 0.408 | 0.676 | 0.361 | +0.024 | no |
| saves | 7569 | +0.021 | 0.048 | 0.200 | 0.717 | +0.016 | no |
| bonus_proxy | 7569 | -0.082 | 0.358 | 0.712 | 0.151 | -0.002 | yes |
| goals_conceded | 7569 | +0.109 | 0.242 | 0.406 | 0.528 | -0.009 | no |
| cards | 7569 | -0.021 | 0.244 | 0.396 | 0.038 | +0.002 | yes |

## Deciles of the predicted score

| decile | n | mean predicted | mean actual |
|---:|---:|---:|---:|
| 1 | 757 | 2.172 | 2.264 |
| 2 | 757 | 2.596 | 2.470 |
| 3 | 757 | 2.855 | 2.898 |
| 4 | 757 | 3.088 | 2.923 |
| 5 | 757 | 3.284 | 3.149 |
| 6 | 756 | 3.463 | 3.237 |
| 7 | 757 | 3.645 | 3.617 |
| 8 | 757 | 3.878 | 3.680 |
| 9 | 757 | 4.251 | 4.017 |
| 10 | 757 | 5.110 | 4.801 |
