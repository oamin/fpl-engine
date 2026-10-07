Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Ninety-minute case study, 2024-25

This table is a case study of five players in 2024-25. It is not a population interval.

The delta is the baseline at the next week minus the points scored in the previous week.
Minutes in the forecast week are not a filter.
The previous week is a single fixture of exactly 90 minutes. A double is excluded.
The three-week columns are blank unless each of those three weeks was also 90 minutes.
Ownership percent is 15 times selected, divided by the gameweek sum of selected.
The five players are the nearest to the 10th, 30th, 50th, 70th, and 90th percentiles of median ownership among players with at least eight such weeks. Ties take the lower id. Points were not used to choose them.
In the qualifying pool of 230 players, the maximum median ownership is Mohamed Salah at 66.7% (with the 95th percentile at 23.8% and 99th at 47.5%), well above the 90th percentile (Kai Havertz, 13.5%); the case study strictly follows the pre-registered percentile lattice and does not sample the extreme ownership tail.
FDR is the official 1–5 difficulty of the player's side in the forecast week.

No reversion term is added to score_xp.

| player | id | 90-minute weeks | median ownership % |
|---|---:|---:|---:|
| Konstantinos Mavropanos | 528 | 16 | 0.1 |
| Mikel Merino | 633 | 10 | 0.6 |
| Ben Johnson | 275 | 11 | 2.2 |
| Martin Ødegaard | 13 | 14 | 5.4 |
| Kai Havertz | 4 | 18 | 13.5 |

## Average of the five

N is how many of the five had a 90-minute previous week. N3 is how many of those rows also have three preceding 90-minute weeks. The three-week mean uses only those rows.

| GW(t+1) | N | N3 | exp_points(t+1) | exp_points(t+1) − actual(t) | score_xp(t+1) | score_xp(t+1) − actual(t) | exp_points(t+1) − mean(t−2:t) | score_xp(t+1) − mean(t−2:t) | FDR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 0 | 5.33 | 0.00 | 3.91 | -1.43 |  |  | 3.7 |
| 3 | 4 | 0 | 3.88 | 0.12 | 4.49 | 0.74 |  |  | 3.2 |
| 4 | 2 | 2 | 5.33 | 0.83 | 4.10 | -0.40 | 0.00 | -1.24 | 3.0 |
| 5 | 2 | 2 | 4.50 | 2.50 | 3.06 | 1.06 | 0.67 | -0.78 | 3.5 |
| 6 | 2 | 2 | 3.90 | 2.40 | 4.83 | 3.33 | 1.23 | 2.17 | 2.0 |
| 7 | 1 | 1 | 5.33 | -0.67 | 7.93 | 1.93 | 2.00 | 4.60 | 1.0 |
| 8 | 2 | 1 | 3.14 | -0.86 | 3.52 | -0.48 | 0.38 | 0.39 | 3.0 |
| 9 | 1 | 1 | 5.25 | 3.25 | 4.57 | 2.57 | -0.08 | -0.76 | 5.0 |
| 10 | 2 | 1 | 3.23 | -1.77 | 3.93 | -1.07 | 0.89 | 1.05 | 4.0 |
| 11 | 3 | 1 | 2.23 | 0.90 | 2.73 | 1.39 | 2.83 | 2.85 | 3.3 |
| 12 | 3 | 1 | 2.03 | -0.97 | 3.12 | 0.12 | 2.85 | 4.21 | 3.3 |
| 13 | 1 | 0 | 1.60 | -1.40 | 3.15 | 0.15 |  |  | 2.0 |
| 14 | 1 | 0 | 4.00 | -2.00 | 5.09 | -0.91 |  |  | 2.0 |
| 15 | 2 | 0 | 2.57 | 1.07 | 3.36 | 1.86 |  |  | 3.0 |
| 16 | 2 | 0 | 0.93 | -0.57 | 1.66 | 0.16 |  |  | 3.0 |
| 17 | 2 | 1 | 2.59 | 0.59 | 3.60 | 1.60 | -0.29 | 1.15 | 3.0 |
| 18 | 1 | 0 | 2.24 | 0.24 | 4.24 | 2.24 |  |  | 2.0 |
| 19 | 3 | 0 | 2.69 | -3.31 | 3.54 | -2.46 |  |  | 3.7 |
| 20 | 1 | 0 | 1.58 | 1.58 | 1.58 | 1.58 |  |  | 4.0 |
| 21 | 1 | 0 | 1.94 | -2.06 | 3.37 | -0.63 |  |  | 2.0 |
| 22 | 3 | 0 | 2.51 | -0.83 | 3.46 | 0.13 |  |  | 3.0 |
| 23 | 4 | 0 | 2.17 | -0.33 | 3.21 | 0.71 |  |  | 3.5 |
| 24 | 1 | 1 | 3.65 | 1.65 | 4.12 | 2.12 | 0.32 | 0.79 | 4.0 |
| 26 | 2 | 0 | 1.58 | -0.92 | 3.01 | 0.51 |  |  | 2.0 |
| 27 | 2 | 0 | 2.38 | 0.38 | 2.80 | 0.80 |  |  | 4.0 |
| 28 | 1 | 0 | 2.28 | -0.72 | 3.28 | 0.28 |  |  | 3.0 |
| 29 | 2 | 1 | 2.40 | -0.10 | 3.45 | 0.95 | -0.06 | 0.89 | 3.0 |
| 30 | 2 | 1 | 1.90 | -3.60 | 2.81 | -2.69 | -2.15 | -1.26 | 3.0 |
| 31 | 4 | 1 | 1.82 | -1.93 | 2.84 | -0.91 | -4.21 | -3.82 | 3.0 |
| 32 | 2 | 1 | 1.70 | 0.20 | 2.46 | 0.96 | -4.24 | -3.65 | 3.5 |
| 33 | 2 | 1 | 1.12 | -5.38 | 2.95 | -3.55 | -4.00 | -3.07 | 3.0 |
| 34 | 1 | 1 | 0.97 | 0.97 | 1.31 | 1.31 | -3.36 | -3.03 | 4.0 |
| 35 | 1 | 0 | 1.21 | 1.21 | 3.30 | 3.30 |  |  | 2.0 |
| 36 | 1 | 0 | 2.80 | -5.20 | 3.05 | -4.95 |  |  | 5.0 |
| 37 | 1 | 0 | 2.86 | -2.14 | 3.61 | -1.39 |  |  | 4.0 |
| 38 | 1 | 1 | 2.95 | -3.05 | 4.35 | -1.65 | -3.39 | -1.98 | 1.0 |

## Ben Johnson

Ben Johnson is the player nearest the 50th percentile of median ownership, at 2.2 percent.

| GW(t+1) | exp_points(t+1) | exp_points(t+1) − actual(t) | score_xp(t+1) | score_xp(t+1) − actual(t) | exp_points(t+1) − mean(t−2:t) | score_xp(t+1) − mean(t−2:t) | FDR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1.50 | -1.50 | 3.60 | 0.60 |  |  | 3.0 |
| 8 | 0.57 | 0.57 | 1.32 | 1.32 |  |  | 3.0 |
| 11 | 0.60 | -1.40 | 1.01 | -0.99 |  |  | 3.0 |
| 12 | 0.64 | -0.36 | 2.05 | 1.05 |  |  | 2.0 |
| 16 | 0.53 | -0.47 | 1.03 | 0.03 |  |  | 3.0 |
| 23 | 0.59 | 1.59 | 1.07 | 2.07 |  |  | 5.0 |
| 26 | 0.64 | -1.36 | 1.95 | -0.05 |  |  | 2.0 |
| 31 | 0.63 | -1.37 | 1.74 | -0.26 |  |  | 3.0 |
| 32 | 0.65 | -0.35 | 1.56 | 0.56 |  |  | 4.0 |
| 33 | 1.00 | -11.00 | 1.93 | -10.07 | -4.00 | -3.07 | 5.0 |
| 34 | 0.97 | 0.97 | 1.31 | 1.31 | -3.36 | -3.03 | 4.0 |

Gemini kept the display rules and reviewed the filled table ([ninety-minute table](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).
