Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Per-90 residual z-score

Per-90 rates divide a minutes-inclusive forecast by the realized minutes of that appearance.
This is a post-match diagnostic, not a pre-deadline decision.
The forward residual is missing when that later week is not a single fixture of at least 30 minutes.
A bar is cleared only when the whole interval is past it.

The residual is points per 90 minus the baseline per 90. The z-score compares that residual with the previous five active appearances in the same season. A week under 30 minutes does not occupy a slot. A double is excluded. The current residual is not inside the mean or the standard deviation. A standard deviation of zero leaves the z-score missing.

Oversold means the z-score is below -1.5. Overbought means it is above +1.5. The value 1.5 itself is in neither bucket.
The forward horizons are the next one, two, and three sheet gameweeks. They were locked before these numbers were read. Where 2022-23 gameweek 7 is absent, the next sheet week after gameweek 6 is gameweek 8.
The Spearman bar is -0.15. The bucket bar is +0.5 per-90 points, oversold minus overbought. The Hurst bar is a median below 0.5. The draw resamples gameweeks inside each season, 1000 times, seed 0. A season with fewer than 20 signal weeks is omitted from the pool. Hurst resamples player-seasons and does not reorder a series. Short-series rescaled range is biased toward 0.5.
The targets were not used to choose the lookback.

No reversion term is added to score_xp.
No winner is declared between score_xp and score_exp_points.

## Spearman of the z-score with the forward residual

| baseline | horizon | correlation | reading |
|---|---:|---|---|
| expected points | 1 | +0.0100 [-0.0079, +0.0282] | the bar is not cleared |
| expected points | 2 | +0.0117 [-0.0069, +0.0285] | the bar is not cleared |
| expected points | 3 | +0.0096 [-0.0100, +0.0277] | the bar is not cleared |
| score_xp | 1 | +0.0339 [+0.0175, +0.0528] | the bar is not cleared |
| score_xp | 2 | -0.0037 [-0.0209, +0.0136] | the bar is not cleared |
| score_xp | 3 | -0.0018 [-0.0195, +0.0160] | the bar is not cleared |

## Oversold minus overbought

| baseline | horizon | oversold | overbought | difference | N oversold | N overbought | reading |
|---|---:|---:|---:|---|---:|---:|---|
| expected points | 1 | +0.59 | +0.51 | +0.0808 [-0.2391, +0.3990] | 968 | 2849 | the bar is not cleared |
| expected points | 2 | +0.53 | +0.38 | +0.1512 [-0.1340, +0.4222] | 934 | 2581 | the bar is not cleared |
| expected points | 3 | +0.36 | +0.52 | -0.1568 [-0.4838, +0.1745] | 889 | 2320 | the bar is not cleared |
| score_xp | 1 | +0.12 | +0.37 | -0.2520 [-0.5260, +0.0161] | 1158 | 2770 | the bar is not cleared |
| score_xp | 2 | +0.13 | +0.03 | +0.0998 [-0.1425, +0.3437] | 1125 | 2516 | the bar is not cleared |
| score_xp | 3 | +0.13 | +0.25 | -0.1224 [-0.4049, +0.1631] | 1050 | 2265 | the bar is not cleared |

## Hurst exponent of the active residual

A series needs 16 finite residuals and two dyadic lags, so the shortest series that enters is 32 appearances. The lag set is powers of two from 8 up to half the length.

| baseline | series | median | share below 0.5 | reading |
|---|---:|---|---:|---|
| expected points | 234 | +0.6382 [+0.6143, +0.6768] | 0.192 | the bar is not cleared |
| score_xp | 234 | +0.6526 [+0.6280, +0.6763] | 0.244 | the bar is not cleared |

## By season

These are point estimates. They are not a second interval.

### expected points

| season | horizon | spearman | bucket difference | signal weeks |
|---|---:|---:|---:|---:|
| 2022-23 | 1 | +0.0005 | +0.1139 | 31 |
| 2023-24 | 1 | -0.0179 | +0.0845 | 32 |
| 2024-25 | 1 | +0.0179 | +0.1491 | 32 |
| 2025-26 | 1 | +0.0316 | -0.0013 | 32 |
| 2022-23 | 2 | +0.0157 | +0.3730 | 30 |
| 2023-24 | 2 | -0.0095 | +0.1476 | 31 |
| 2024-25 | 2 | +0.0342 | -0.3418 | 31 |
| 2025-26 | 2 | +0.0035 | +0.4512 | 31 |
| 2022-23 | 3 | +0.0081 | +0.4386 | 29 |
| 2023-24 | 3 | +0.0073 | -0.5174 | 30 |
| 2024-25 | 3 | +0.0069 | -0.1922 | 30 |
| 2025-26 | 3 | +0.0150 | -0.2938 | 30 |

### score_xp

| season | horizon | spearman | bucket difference | signal weeks |
|---|---:|---:|---:|---:|
| 2022-23 | 1 | +0.0406 | -0.3668 | 31 |
| 2023-24 | 1 | +0.0313 | -0.1923 | 32 |
| 2024-25 | 1 | +0.0308 | -0.3835 | 32 |
| 2025-26 | 1 | +0.0384 | -0.0520 | 32 |
| 2022-23 | 2 | +0.0027 | +0.0450 | 30 |
| 2023-24 | 2 | -0.0036 | -0.1228 | 31 |
| 2024-25 | 2 | +0.0187 | -0.2172 | 31 |
| 2025-26 | 2 | -0.0275 | +0.6164 | 31 |
| 2022-23 | 3 | +0.0080 | +0.1399 | 29 |
| 2023-24 | 3 | -0.0078 | -0.4938 | 30 |
| 2024-25 | 3 | -0.0035 | -0.0748 | 30 |
| 2025-26 | 3 | +0.0015 | -0.0578 | 30 |

No bar is cleared.
The one-week score_xp correlation stays above zero.
Both Hurst intervals stay above 0.5.
Every bucket interval covers zero, so a gap of half a point per 90 is not established.
The 2025-26 horizon-2 point is not the pool.
No reversion term is added to score_xp.

Gemini kept the formula and reviewed the metrics ([reversion signal](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).
