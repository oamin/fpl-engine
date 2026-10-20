# L4 chip half-stopping vs L4 no-chip

Priced t:t+2, γ=1. `chip_stop` = TC/BB leave-one-out DP τ(r) + WC≥16 / FH≥12.

## Pooled

| arm | pts | hit pts |
| --- | ---: | ---: |
| l4 | 8027 | 284 |
| chip_stop | 8004 | 228 |

Chip plays: `{'triple_captain': 8, 'free_hit': 7, 'bench_boost': 5, 'wildcard': 3}`

## By season

| season | L4 | stop | L4h | sh | chip weeks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 2038 | 2054 | 80 | 72 | 6 |
| 2023-24 | 1946 | 1885 | 68 | 56 | 5 |
| 2024-25 | 2029 | 2052 | 92 | 68 | 7 |
| 2025-26 | 2014 | 2013 | 44 | 32 | 5 |

## chip_stop − l4

- Mean Δ/GW: **-0.185** [-2.146, +1.775]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +0.516 |
| 2023-24 | -1.968 |
| 2024-25 | +0.742 |
| 2025-26 | -0.032 |

## Verdict

Promote only if paired interval clears 0 vs L4. Else park.

JSON: `data/processed/layer4_chip_stop.json`
