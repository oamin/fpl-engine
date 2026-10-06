# Outside expected points on the locked cohort, Gameweeks 1–5

The question is whether Onside or the official `ep_next` scores the players in the veterans' squads, and in the rank slots' squads, more accurately than `score_xp`. The climb is not rerun. `score_xp` stays the published score.

The 14 managers were locked on 4 October. Seven are the veterans and seven are the rank slots. ojaminFC stays out of this count. A player-week is one owned player in one gameweek. The same player in several squads counts once in the unique table and once per squad in the weighted table. Accuracy is absolute error against the official points, on rows with minutes above 0.

The outside file is the Onside Arena graded CSV, CC-BY-4.0, with official `ep_next` on the same rows. Cite Onside Arena and [10.5281/zenodo.22746985](https://doi.org/10.5281/zenodo.22746985). Every capture in that file is before its deadline. Gameweek 2 was captured the evening of Gameweek 1, Onside has 13 values capped at 2.95, and `ep_next` copies Gameweek 1. That week stays out of the tables below. The joined rows are `data/processed/cohort_xp_gw15.csv`.

## The cut

Gameweeks 1, 3, 4, and 5. Minutes above 0. A published score and both outside numbers present.

`score_xp` is the feature-row score. Twelve of the 237 unique rows use the early score instead, the past-only value capped at 6 for a player with one or two prior appearances. Dropping those twelve leaves the order unchanged.

Ten Gameweek 1 appearances have no published score, because the player has no prior appearance: Rushworth, Thomas, van Ewijk, McBurnie, O'Shea, Davis, Walle Egeli, Tzolis, Scherpen, and M.Sangaré. Sangaré scored 14. Those ten stay out of the error, since one side has no number. Pedro Porro is missing from Onside in Gameweek 2, which is already out.

## Error

Unique player-weeks:

| Group | Rows | score_xp | Onside | ep_next | Points | MAE score_xp | MAE Onside | MAE ep_next |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Veterans | 149 | 4.15 | 3.46 | 3.62 | 4.44 | 3.00 | 3.13 | 3.22 |
| Rank slots | 191 | 3.95 | 3.25 | 3.42 | 4.73 | 3.07 | 3.14 | 3.36 |

Bias, forecast minus points: veterans −0.29, −0.99, −0.82. Rank slots −0.79, −1.48, −1.31.

`score_xp` is the closer number on 83 of 149 veteran rows, and Onside on 66. On the rank slots, `score_xp` is closer on 101 of 191, and Onside on 90. `ep_next` is closer than `score_xp` on 70 veteran rows and 86 rank rows.

Counting a player once for each squad that owned him gives the same order. Veterans, 395 rows: MAE 3.26, 3.42, 3.56. Rank slots, 386 rows: MAE 3.38, 3.44, 3.59.

The stored eleven, still unique and still with minutes above 0: veterans 109 rows, MAE 3.25, 3.36, 3.54. Rank slots 144 rows, MAE 3.26, 3.38, 3.62.

## Where the outside number wins a week

On the union of the two groups, one week at a time:

| GW | Rows | MAE score_xp | MAE Onside | MAE ep_next |
|---|---:|---:|---:|---:|
| 1 | 47 | 3.24 | 3.16 | 3.12 |
| 3 | 68 | 2.39 | 2.50 | 2.81 |
| 4 | 61 | 3.37 | 3.58 | 3.86 |
| 5 | 61 | 3.09 | 3.16 | 3.34 |

Gameweek 1 is the week the outside numbers are closer. The other three weeks go to `score_xp`. The rank-only players, the 88 who were not also in a veteran squad, are the near-tie: MAE 2.97 against Onside 2.99, and Onside is closer on 46 rows to 42.

## Reading

Onside and `ep_next` do not score these squads more accurately than `score_xp`. Both sit further below the points. The rank slots' players scored 4.73 against a `score_xp` of 3.95, an Onside value of 3.25, and an `ep_next` of 3.42. The gap to those managers is players beating every one of these forecasts. The outside forecasts are the ones further away.

The published total stays 280. These columns are not a new score. Gemini kept the reading ([cohort xp](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)). Gameweek 2 stays out, and the ten Gameweek 1 debuts with no published score stay out.
