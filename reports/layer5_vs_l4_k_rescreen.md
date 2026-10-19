# Layer 5 beam K grid vs L4 priced — horizon-MILP re-screen

K ∈ 1, 2, 4, 8, 16. L4 = priced t:t+2 no chips (γ=1). L5 = joint chip beam; depth-0 = `solve_horizon_transfers`. time_limit = 45.0s.

Promote a K only if L5(K)−L4 interval is clear of 0 above.

## Paired vs L4 by K

### K=1

- Mean Δ/GW: **-0.161** [-2.194, +2.057]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -0.968 |
| 2023-24 | -0.645 |
| 2024-25 | +3.032 |
| 2025-26 | -2.065 |

### K=2

- Mean Δ/GW: **+0.113** [-1.855, +2.105]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -1.323 |
| 2023-24 | +1.677 |
| 2024-25 | -0.194 |
| 2025-26 | +0.290 |

### K=4

- Mean Δ/GW: **+0.323** [-1.274, +1.976]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +0.806 |
| 2023-24 | -1.194 |
| 2024-25 | +1.774 |
| 2025-26 | -0.097 |

### K=8

- Mean Δ/GW: **+0.734** [-0.847, +2.355]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +1.032 |
| 2023-24 | -1.194 |
| 2024-25 | +1.774 |
| 2025-26 | +1.323 |

### K=16

- Mean Δ/GW: **+0.734** [-0.847, +2.355]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +1.032 |
| 2023-24 | -1.194 |
| 2024-25 | +1.774 |
| 2025-26 | +1.323 |

## Season totals

| season | L4 | K=1 | K=2 | K=4 | K=8 | K=16 | ΔK=1 | ΔK=2 | ΔK=4 | ΔK=8 | ΔK=16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 2038 | 2008 | 1997 | 2063 | 2070 | 2070 | -30 | -41 | +25 | +32 | +32 |
| 2023-24 | 1946 | 1926 | 1998 | 1909 | 1909 | 1909 | -20 | +52 | -37 | -37 | -37 |
| 2024-25 | 2029 | 2123 | 2023 | 2084 | 2084 | 2084 | +94 | -6 | +55 | +55 | +55 |
| 2025-26 | 2014 | 1950 | 2023 | 2011 | 2055 | 2055 | -64 | +9 | -3 | +41 | +41 |

**Pooled Δ vs L4:** K=1 **-20** · K=2 **+14** · K=4 **+40** · K=8 **+91** · K=16 **+91**

## Verdict

K=1 **inconclusive** vs L4.

K=2 **inconclusive** vs L4.

K=4 **inconclusive** vs L4.

K=8 **inconclusive** vs L4.

K=16 **inconclusive** vs L4.

Best pooled point estimate: **K=8** (Δ +91). **PI lock 2026-10-16: `DEFAULT_BEAM_K=8`** (search budget; L4 remains strategiser default — L5 not promoted).

JSON: `data/processed/layer5_vs_l4_k_rescreen.json`
