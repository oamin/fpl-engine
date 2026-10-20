# L4 hit controls — budget + gate vs unconstrained

Priced t:t+2, γ=1. Baseline `l4` = locked unconstrained hits.

| arm | rule |
| --- | --- |
| `l4` | unconstrained (max_hits=15) |
| `budget` | season hit_points ≤ 16 (≤4 hits) |
| `gate` | objective hit penalty = 4+4 = 8; realised still −4/hit |

## Hits + FT (pooled)

| arm | pts | hit pts |
| --- | ---: | ---: |
| l4 | 8027 | 284 |
| budget | 7871 | 64 |
| gate | 7897 | 48 |

## By season

| season | L4 pts | L4 hits | bud pts | bud hits | gate pts | gate hits | mean FT L4 | mean FT bud | mean FT gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 2038 | 80 | 1992 | 16 | 2065 | 16 | 1.06 | 1.06 | 1.06 |
| 2023-24 | 1946 | 68 | 1930 | 16 | 1953 | 4 | 1.06 | 1.06 | 1.06 |
| 2024-25 | 2029 | 92 | 2045 | 16 | 1990 | 16 | 1.06 | 1.06 | 1.06 |
| 2025-26 | 2014 | 44 | 1904 | 16 | 1889 | 12 | 1.06 | 1.06 | 1.06 |

## Points vs L4

### budget − l4

- Mean Δ/GW: **-1.258** [-2.927, +0.428]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -1.484 |
| 2023-24 | -0.516 |
| 2024-25 | +0.516 |
| 2025-26 | -3.548 |

### gate − l4

- Mean Δ/GW: **-1.048** [-2.880, +0.871]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +0.871 |
| 2023-24 | +0.226 |
| 2024-25 | -1.258 |
| 2025-26 | -4.032 |

## Verdict

Controls are baselines for the FT-DP arm — not promote candidates.
Prefer lower hits / higher mean FT without a collapse in points.

JSON: `data/processed/layer4_hit_controls.json`
