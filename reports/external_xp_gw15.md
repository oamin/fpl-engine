# Outside expected points on the Gameweeks 1–5 squad

The published climb scores 280. That figure is the points the squad returned after automatic substitutes, with the captain doubled and no hits. It is Gameweeks 40, 89, 48, 62, and 41. This note places two outside pre-deadline numbers next to `score_xp` on those same players. The climb is not rerun. `score_xp` stays the published score.

## What was available

No open-source model has a frozen 2026/27 Gameweeks 1–5 file. OpenFPL is MIT-licensed and its published test is the 2024/25 season. Fitting it now on weeks that have already been played would be a hindsight run. The Vaastav `xP` column is the official `ep_this`, scraped after the gameweek. The 177arc expected-points file has no capture time, so a past week in it cannot be shown to have been frozen before the deadline.

The file that does carry a capture time is the Onside Arena graded-predictions CSV, CC-BY-4.0. The model code was not found as an open repository. Each row also stores the official FPL `ep_next` from the same capture. Cite Onside Arena and [10.5281/zenodo.22746985](https://doi.org/10.5281/zenodo.22746985). The current file is [graded-predictions.csv](https://onsidearena.com/data/graded-predictions.csv). The squad slice is `data/processed/external_xp_gw15.csv`. The full third-party file is not in the repo.

The download used here has 2,982 rows, Gameweeks 1–6, model version v5. Every row has `captured_at` earlier than `deadline_utc`.

| GW | Deadline (UTC) | Captured (UTC) |
|---|---|---|
| 1 | 2026-08-21 17:30 | 2026-08-21 15:00 |
| 2 | 2026-08-28 17:30 | 2026-08-21 18:50 |
| 3 | 2026-09-04 17:30 | 2026-08-28 22:12 |
| 4 | 2026-09-12 12:30 | 2026-09-04 22:19 |
| 5 | 2026-09-18 17:30 | 2026-09-12 22:12 |

Gameweek 2 was captured about an hour after the Gameweek 1 deadline, a week before its own deadline. Onside has 13 distinct values that week and a maximum of 2.95. Official `ep_next` on those rows matches the Gameweek 1 value for 501 of the 502 players who appear in both weeks. The one change is Kudus, 1.7 to 0.6. Gameweek 2 stays in the player table and stays out of the averages below.

Gameweek 1 gives Dasilva, element 103, an Onside value of 14.99 against 0 points and an `ep_next` of 1.5. He is not in this squad. That row is not in the averages.

## Join

The squad file is `data/processed/own_squad_gw15.csv`. The element id is the number after `2026-27:`. 73 of 75 rows match. Sánchez, element 140, is absent from Onside in Gameweeks 4 and 5, after the Como loan. Element 573 is Juanlu Sánchez and is not joined. Points on the 73 matched rows agree. Every matched capture is before that week's deadline.

## Level

On the 55 squad rows with minutes above 0, leaving Gameweek 2 out:

| | score_xp | Onside | ep_next | Points |
|---|---:|---:|---:|---:|
| Mean | 4.52 | 3.34 | 2.81 | 3.85 |
| Mean absolute error | 2.99 | 2.67 | 2.67 | |
| Bias against points | +0.67 | −0.51 | −1.04 | |

`score_xp` is above Onside on 48 of these 55 rows, and above `ep_next` on 46 of 55. Including the Gameweek 2 appearances, it is above Onside on 61 of 68.

The named eleven, with the captain counted twice, on players Onside still lists:

| GW | score_xp | Onside | ep_next | Named points | Week total |
|---|---:|---:|---:|---:|---:|
| 1 | 62.14 | 42.86 | 34.80 | 40 | 40 |
| 2 | 60.18 | 25.25 | 34.90 | 79 | 89 |
| 3 | 59.73 | 41.59 | 28.00 | 45 | 48 |
| 4 | 49.17 | 39.00 | 35.90 | 54 | 62 |
| 5 | 55.42 | 48.68 | 46.00 | 35 | 41 |

Gameweeks 4 and 5 omit Sánchez from all three projection columns in that sum, because Onside has no row. His `score_xp` is still 3.61 and his points are 0, so the week total is unchanged by leaving him out of the sum. The named-points column is the eleven before automatic substitutes. The week total is the published 280's weekly piece, after those substitutes. Gameweek 2's outside columns are the stale grid.

On Gameweeks 1, 3, 4, and 5 the captain-doubled named sums are 226.46, 172.13, and 144.70. The named points in those four weeks are 174. The week totals are 191. Onside's sum sits next to the named points. `score_xp` sits above them, because the captain's projection is doubled and several of those weeks were quiet. That is a high projection of this eleven, not a low one.

## Haaland, for scale

He is not in the fifteen. His published scores are 6.34, 5.79, 7.18, 5.15, and 6.47. Onside is 4.61, 1.49, 5.88, 3.63, and 6.06. `ep_next` is 4.0, 4.0, 7.5, 5.0, and 6.0. The Gameweek 2 Onside figure is the capped grid. On the four usable weeks (1, 3, 4, and 5) his points were 26, published scores sum to 25.14, Onside sums to 20.18, and `ep_next` sums to 22.50 (across all five weeks, points are 39, published scores sum to 30.93, and Onside sums to 21.67, where 1.49 is the Gameweek 2 cap). The outside file does not show a Haaland projection this squad's values somehow missed.

## Reading

The 280 is what this fifteen returned. On the same names, `score_xp` is the higher of the three pre-deadline columns in almost every usable row. An outside freeze of these five weeks does not describe the squad as under-priced. It prices the same players lower, and the points landed between `score_xp` and those lower figures. The published total stays 280. Gemini kept the reading ([external xp](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)). The Gameweek 2 grid stays out of the averages, and Haaland's four usable weeks are the scale check.

### Gameweek 1

Solved from the buy pool. Week total 40. No automatic substitute.

| Player | Pos | Lineup | score_xp | Onside | ep_next | Points | Minutes |
|---|---|---|---:|---:|---:|---:|---:|
| B.Fernandes (C) | MID | XI | 6.77 | 4.47 | 4.00 | 2 | 90 |
| Mbeumo (V) | MID | XI | 5.73 | 4.05 | 2.80 | 2 | 90 |
| Gabriel | DEF | XI | 5.60 | 4.12 | 4.00 | 5 | 90 |
| Guéhi | DEF | XI | 5.19 | 3.43 | 2.80 | 10 | 90 |
| O'Reilly | DEF | XI | 5.12 | 3.86 | 3.10 | 2 | 62 |
| Thiago | FWD | XI | 4.89 | 3.57 | 2.50 | 0 | 82 |
| Hincapie | DEF | XI | 4.73 | 3.44 | 2.50 | 1 | 9 |
| Anderson | MID | XI | 4.70 | 2.56 | 2.30 | 2 | 62 |
| Schade | MID | XI | 4.55 | 3.44 | 2.10 | 3 | 90 |
| Ndiaye | MID | XI | 4.48 | 2.98 | 2.10 | 9 | 90 |
| Sánchez | GKP | XI | 3.61 | 2.47 | 2.60 | 2 | 90 |
| Colwill | DEF | bench | 4.45 | 2.47 | 2.20 | 0 | 90 |
| Šeško | FWD | bench | 3.98 | 3.69 | 1.70 | 1 | 23 |
| Calvert-Lewin | FWD | bench | 3.68 | 3.33 | 2.00 | 1 | 90 |
| Verbruggen | GKP | bench | 3.61 | 2.65 | 1.90 | 6 | 90 |

### Gameweek 2

No transfers. Week total 89. Two automatic substitutes. The Onside column this week is the 13-value grid, and ep_next matches Gameweek 1 on 501 of 502 players in the source file.

| Player | Pos | Lineup | score_xp | Onside | ep_next | Points | Minutes |
|---|---|---|---:|---:|---:|---:|---:|
| B.Fernandes (C) | MID | XI | 6.87 | 1.86 | 4.00 | 23 | 90 |
| Mbeumo (V) | MID | XI | 6.10 | 1.86 | 2.80 | 11 | 90 |
| Guéhi | DEF | XI | 5.06 | 2.37 | 2.80 | 2 | 90 |
| Gabriel | DEF | XI | 5.06 | 2.06 | 4.00 | 8 | 90 |
| O'Reilly | DEF | XI | 4.79 | 2.37 | 3.10 | 2 | 88 |
| Hincapie | DEF | XI | 4.73 | 2.06 | 2.50 | 0 | 0 |
| Anderson | MID | XI | 4.54 | 1.71 | 2.30 | 3 | 81 |
| Thiago | FWD | XI | 4.29 | 1.49 | 2.50 | 2 | 90 |
| Colwill | DEF | XI | 4.13 | 2.95 | 2.20 | 1 | 71 |
| Ndiaye | MID | XI | 4.13 | 1.71 | 2.10 | 4 | 90 |
| Sánchez | GKP | XI | 3.61 | 2.95 | 2.60 | 0 | 0 |
| Schade | MID | bench | 4.07 | 1.71 | 2.10 | 10 | 87 |
| Calvert-Lewin | FWD | bench | 3.89 | 1.49 | 2.00 | 8 | 90 |
| Šeško | FWD | bench | 3.77 | 1.49 | 1.70 | 1 | 10 |
| Verbruggen | GKP | bench | 3.28 | 2.06 | 1.90 | 0 | 90 |

### Gameweek 3

In Van Hecke and Thiaw, out O'Reilly and Hincapie. Week total 48. One automatic substitute.

| Player | Pos | Lineup | score_xp | Onside | ep_next | Points | Minutes |
|---|---|---|---:|---:|---:|---:|---:|
| Guéhi (C) | DEF | XI | 5.86 | 8.61 | 6.00 | 8 | 90 |
| B.Fernandes (V) | MID | XI | 5.77 | 1.90 | 1.00 | 2 | 90 |
| Thiago | FWD | XI | 5.15 | 3.74 | 0.00 | 2 | 90 |
| Ndiaye | MID | XI | 5.10 | 2.55 | 4.50 | 3 | 86 |
| Mbeumo | MID | XI | 5.09 | 2.90 | 1.00 | 8 | 90 |
| Anderson | MID | XI | 4.93 | 3.48 | 2.50 | 3 | 90 |
| Gabriel | DEF | XI | 4.80 | 1.79 | 2.50 | 2 | 90 |
| Schade | MID | XI | 4.71 | 2.93 | 1.50 | 2 | 90 |
| Thiaw | DEF | XI | 4.54 | 1.92 | 1.50 | -1 | 90 |
| Van Hecke | DEF | XI | 4.31 | 1.82 | 0.50 | 8 | 90 |
| Sánchez | GKP | XI | 3.61 | 1.34 | 1.00 | 0 | 0 |
| Colwill | DEF | bench | 4.13 | 1.38 | 0.00 | 0 | 0 |
| Verbruggen | GKP | bench | 3.61 | 2.19 | 3.00 | 3 | 90 |
| Calvert-Lewin | FWD | bench | 3.48 | 1.41 | 0.50 | 1 | 72 |
| Šeško | FWD | bench | 2.35 | 0.39 | 0.50 | 5 | 20 |

### Gameweek 4

No transfers. Week total 62. One automatic substitute. Sánchez has no Onside row.

| Player | Pos | Lineup | score_xp | Onside | ep_next | Points | Minutes |
|---|---|---|---:|---:|---:|---:|---:|
| Gabriel (C) | DEF | XI | 4.95 | 2.66 | 4.30 | 9 | 90 |
| Van Hecke (V) | DEF | XI | 4.81 | 2.75 | 0.70 | 8 | 90 |
| B.Fernandes | MID | XI | 4.80 | 6.01 | 8.30 | 2 | 90 |
| Colwill | DEF | XI | 4.57 | 2.81 | 0.30 | 1 | 90 |
| Anderson | MID | XI | 4.28 | 2.08 | 1.70 | 5 | 90 |
| Calvert-Lewin | FWD | XI | 4.27 | 3.61 | 3.00 | 10 | 77 |
| Guéhi | DEF | XI | 4.24 | 4.19 | 4.00 | 6 | 90 |
| Mbeumo | MID | XI | 4.22 | 5.77 | 4.30 | 2 | 90 |
| Thiago | FWD | XI | 4.05 | 3.99 | 0.70 | 1 | 90 |
| Ndiaye | MID | XI | 4.03 | 2.47 | 4.30 | 1 | 45 |
| Sánchez | GKP | XI | 3.61 | — | — | 0 | 0 |
| Thiaw | DEF | bench | 3.91 | 2.29 | 3.00 | 2 | 90 |
| Schade | MID | bench | 3.79 | 3.66 | 4.30 | 15 | 90 |
| Verbruggen | GKP | bench | 3.60 | 2.57 | 2.00 | 8 | 90 |
| Šeško | FWD | bench | 2.02 | 0.38 | 0.70 | 1 | 23 |

### Gameweek 5

No transfers. Week total 41. One automatic substitute. Sánchez has no Onside row.

| Player | Pos | Lineup | score_xp | Onside | ep_next | Points | Minutes |
|---|---|---|---:|---:|---:|---:|---:|
| B.Fernandes (C) | MID | XI | 5.64 | 5.75 | 6.80 | 2 | 90 |
| Guéhi (V) | DEF | XI | 5.41 | 5.82 | 5.00 | 4 | 90 |
| Thiaw | DEF | XI | 5.18 | 3.20 | 2.00 | 4 | 90 |
| Mbeumo | MID | XI | 5.01 | 5.58 | 5.20 | 2 | 90 |
| Calvert-Lewin | FWD | XI | 4.97 | 3.75 | 2.50 | 2 | 90 |
| Van Hecke | DEF | XI | 4.96 | 4.54 | 4.50 | 6 | 90 |
| Gabriel | DEF | XI | 4.84 | 4.42 | 6.00 | 1 | 90 |
| Anderson | MID | XI | 4.67 | 2.66 | 2.00 | 2 | 90 |
| Ndiaye | MID | XI | 4.65 | 2.95 | 4.00 | 5 | 74 |
| Thiago | FWD | XI | 4.45 | 4.26 | 1.20 | 5 | 90 |
| Sánchez | GKP | XI | 3.61 | — | — | 0 | 0 |
| Schade | MID | bench | 4.11 | 4.48 | 7.50 | 9 | 90 |
| Verbruggen | GKP | bench | 3.26 | 2.45 | 2.20 | 6 | 90 |
| Colwill | DEF | bench | 3.25 | 2.52 | 0.50 | 1 | 90 |
| Šeško | FWD | bench | 2.02 | 0.48 | 1.80 | 0 | 0 |
