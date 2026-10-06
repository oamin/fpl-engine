# Queued tests

Both tests were locked before any total was read. The score and the search are unchanged. Gemini kept both calls ([queued tests](bc-7121b96b-db3a-552d-ade8-335c01d37eda)).

## Search shadow

The shadow is the stage 44 count: the three-week value of the best cross-position pair the beam does not try, minus the move it chose. A week with no clubs is skipped, as on the published climb. 2025/26 was already 0.06 a week, across 33 weeks, with no week at 1.

A rewrite was to be rejected if each of 2022/23, 2023/24, and 2024/25 averaged under 0.25 and had fewer than three weeks at or above 1.

| season | weeks | mean value | weeks at or above 1 | largest week |
|---|---:|---:|---:|---:|
| 2022/23 | 32 | 0.005 | 0 | 0.17 |
| 2023/24 | 33 | 0.004 | 0 | 0.12 |
| 2024/25 | 33 | 0.101 | 1 | 1.14 |

Every season is under both bars. Across the four seasons, one week reaches 1, and that week is 1.14. The rewrite is rejected. The search stays. See `reports/search_shadow.md`.

## Clean fill

The fast eleven uses the previous season as the only fill. A matched player keeps last season's shifted prior. A debutant takes the position mean of the matched players' first rows. The total is the eleven plus the captain. The published eleven, Gameweeks 5–8, is the within-season fill.

A season was kept only if Gameweeks 1–8 trailed Gameweeks 9–38 by at least 2 points a week and Gameweeks 5–8 trailed the published eleven by at least 1 point a week. The hypothesis needed two of the three seasons. Every window is complete.

| season | GW1–8 | GW9–38 | early gap | clean GW5–8 | published GW5–8 | published gap | kept |
|---|---:|---:|---:|---:|---:|---:|---|
| 2023/24 | 53.25 | 63.43 | +10.18 | 54.75 | 58.50 | +3.75 | yes |
| 2024/25 | 75.88 | 65.90 | −9.97 | 66.00 | 55.75 | −10.25 | no |
| 2025/26 | 61.75 | 60.93 | −0.82 | 59.25 | 69.00 | +9.75 | no |

2024/25 opens at 88, 107, and 97, so the early weeks are ahead of the rest of that season, and the clean eleven also beats the published one. 2025/26 is level across the season and the clean Gameweeks 5–8 trail the published eleven. One season cleared. Cold start is rejected. The score stays. See `reports/clean_fill.md`.

## Batch

The search shadow is the quieter of the two results: the largest mean is 0.10. Cold start fails the two-season bar. Neither changes the engine. The upper-tail reading stays off the queue.
