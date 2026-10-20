# L4 FT-DP vs hit controls

Priced t:t+2, γ=1. `ft_dp` = leave-one-out DP thresholds on roll vs spend.

## Pooled

| arm | pts | hit pts |
| --- | ---: | ---: |
| l4 | 8027 | 284 |
| budget | 7871 | 64 |
| gate | 7897 | 48 |
| ft_dp | 7955 | 276 |

DP choices: `{'ft': 54, 'hit': 42, 'roll': 24, 'wildcard': 4}`

## By season (pts / hit pts / mean FT)

| season | L4 | bud | gate | dp | L4h | bh | gh | dph | FT L4 | FT dp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 2038 | 1992 | 2065 | 2082 | 80 | 16 | 16 | 72 | 1.06 | 1.06 |
| 2023-24 | 1946 | 1930 | 1953 | 1926 | 68 | 16 | 4 | 72 | 1.06 | 1.06 |
| 2024-25 | 2029 | 2045 | 1990 | 1977 | 92 | 16 | 16 | 80 | 1.06 | 1.06 |
| 2025-26 | 2014 | 1904 | 1889 | 1970 | 44 | 16 | 12 | 52 | 1.06 | 1.06 |

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

### ft_dp − l4

- Mean Δ/GW: **-0.581** [-1.903, +0.863]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +1.419 |
| 2023-24 | -0.645 |
| 2024-25 | -1.677 |
| 2025-26 | -1.419 |

## Verdict

Promote FT-DP only if hits fall toward control levels without a clear points loss vs L4, and mean FT rises (hoarding). Else park.

JSON: `data/processed/layer4_ft_dp.json`
