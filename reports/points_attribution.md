# Points attribution (rulebook)

Share of **aggregate** `total_points` by FPL scoring component. Reconstruction matches `total_points` exactly (resid = 0).

- Season rows: **29757**
- Aggregate points: **34409**

## ALL (Σ points = 34409)

| channel | points | % of total | % of positive mass |
|---|---:|---:|---:|
| appearance | 19313 | 56.1% | 51.2% |
| clean_sheets | 5050 | 14.7% | 13.4% |
| goals | 4843 | 14.1% | 12.8% |
| defcon | 2834 | 8.2% | 7.5% |
| assists | 2826 | 8.2% | 7.5% |
| bonus | 2419 | 7.0% | 6.4% |
| saves | 448 | 1.3% | 1.2% |
| cards_og_pens | -1610 | -4.7% | 0.0% |
| goals_conceded | -1714 | -5.0% | 0.0% |

## played (Σ points = 34410)

| channel | points | % of total | % of positive mass |
|---|---:|---:|---:|
| appearance | 19313 | 56.1% | 51.2% |
| clean_sheets | 5050 | 14.7% | 13.4% |
| goals | 4843 | 14.1% | 12.8% |
| defcon | 2834 | 8.2% | 7.5% |
| assists | 2826 | 8.2% | 7.5% |
| bonus | 2419 | 7.0% | 6.4% |
| saves | 448 | 1.3% | 1.2% |
| cards_og_pens | -1609 | -4.7% | 0.0% |
| goals_conceded | -1714 | -5.0% | 0.0% |

## mins>=60 (Σ points = 29882)

| channel | points | % of total | % of positive mass |
|---|---:|---:|---:|
| appearance | 15630 | 52.3% | 47.7% |
| clean_sheets | 5050 | 16.9% | 15.4% |
| goals | 4124 | 13.8% | 12.6% |
| defcon | 2820 | 9.4% | 8.6% |
| assists | 2388 | 8.0% | 7.3% |
| bonus | 2297 | 7.7% | 7.0% |
| saves | 447 | 1.5% | 1.4% |
| cards_og_pens | -1256 | -4.2% | 0.0% |
| goals_conceded | -1618 | -5.4% | 0.0% |

## GKP (Σ points = 2564)

| channel | points | % of total | % of positive mass |
|---|---:|---:|---:|
| appearance | 1524 | 59.4% | 52.2% |
| clean_sheets | 772 | 30.1% | 26.4% |
| saves | 448 | 17.5% | 15.3% |
| bonus | 162 | 6.3% | 5.5% |
| assists | 15 | 0.6% | 0.5% |
| goals | 0 | 0.0% | 0.0% |
| defcon | 0 | 0.0% | 0.0% |
| cards_og_pens | -11 | -0.4% | 0.0% |
| goals_conceded | -346 | -13.5% | 0.0% |

## DEF (Σ points = 12038)

| channel | points | % of total | % of positive mass |
|---|---:|---:|---:|
| appearance | 6976 | 57.9% | 49.4% |
| clean_sheets | 3288 | 27.3% | 23.3% |
| defcon | 1642 | 13.6% | 11.6% |
| goals | 822 | 6.8% | 5.8% |
| assists | 711 | 5.9% | 5.0% |
| bonus | 676 | 5.6% | 4.8% |
| saves | 0 | 0.0% | 0.0% |
| cards_og_pens | -709 | -5.9% | 0.0% |
| goals_conceded | -1368 | -11.4% | 0.0% |

## MID (Σ points = 15575)

| channel | points | % of total | % of positive mass |
|---|---:|---:|---:|
| appearance | 8612 | 55.3% | 52.8% |
| goals | 2665 | 17.1% | 16.3% |
| assists | 1779 | 11.4% | 10.9% |
| defcon | 1174 | 7.5% | 7.2% |
| bonus | 1098 | 7.0% | 6.7% |
| clean_sheets | 990 | 6.4% | 6.1% |
| goals_conceded | 0 | 0.0% | 0.0% |
| saves | 0 | 0.0% | 0.0% |
| cards_og_pens | -743 | -4.8% | 0.0% |

## FWD (Σ points = 4232)

| channel | points | % of total | % of positive mass |
|---|---:|---:|---:|
| appearance | 2201 | 52.0% | 50.3% |
| goals | 1356 | 32.0% | 31.0% |
| bonus | 483 | 11.4% | 11.0% |
| assists | 321 | 7.6% | 7.3% |
| defcon | 18 | 0.4% | 0.4% |
| clean_sheets | 0 | 0.0% | 0.0% |
| goals_conceded | 0 | 0.0% | 0.0% |
| saves | 0 | 0.0% | 0.0% |
| cards_og_pens | -147 | -3.5% | 0.0% |

## Plots

- `data/plots/points_attribution.png`
- `data/plots/points_attribution_by_pos.png`

## Read

- **% of total** = channel sum / Σ total_points (negatives reduce the total).
- **% of positive mass** = max(channel,0) / sum of positive channel totals — better “where points come from”.
- Appearance = 1pt (1–59′) or 2pts (60′+).
