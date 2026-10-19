# Layer 5 beam (K=2) vs L4 priced — horizon-MILP re-screen

Shared μ_post wildcard open. L4 = priced t:t+2 transfers (γ=1), no chips. L5 = joint chip+transfer beam with **K=2**. Unconstrained hits (−4/hit). Promote L5 only if L5−L4 paired interval clears 0.

Re-screen after depth-0 none/TC/BB = `solve_horizon_transfers` (prior Step 5 was myopic weekly transfers).
time_limit = 45.0s · board = priced hard-stop only.

## Paired GW Δpoints (L5 − L4)

- Mean Δ/GW: **+0.113** [-1.855, +2.105]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -1.323 |
| 2023-24 | +1.677 |
| 2024-25 | -0.194 |
| 2025-26 | +0.290 |

## Season totals

| season | n GW | L4 | L5 | Δ | L4 hits | L5 hits | L5 chip weeks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 31 | 2038 | 1997 | -41 | 80 | 108 | 7 |
| 2023-24 | 31 | 1946 | 1998 | +52 | 68 | 68 | 7 |
| 2024-25 | 31 | 2029 | 2023 | -6 | 92 | 80 | 8 |
| 2025-26 | 31 | 2014 | 2023 | +9 | 44 | 48 | 8 |

**Pooled:** L4 8027 · L5 8041 · Δ **+14**

## Verdict

L5 vs L4 **inconclusive** — keep L4 default; optional live shadow of L5.

JSON: `data/processed/layer5_vs_l4_rescreen.json`
