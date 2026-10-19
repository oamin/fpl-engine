# Layer 4 μ_post priced path — GW(t:t+2) vs L3

Week **t** = Layer-3 `mu_post`. Weeks **t+1, t+2** = projected `score_xp_v2` (frozen priors + FD opening pots) then the same constant-K shrink. Hard stop (no beyond-t+2). Covariance draws not used. Hits unconstrained (solver bound max_hits=15); objective still −4 per hit.

- γ = 0.9; terminal = 0; max_hits = 15

## Paired GW Δpoints (L4 − L3)

- Mean Δ/GW: **+2.548** [+0.411, +4.637]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +4.387 |
| 2023-24 | +3.290 |
| 2024-25 | -1.548 |
| 2025-26 | +4.065 |

## Season totals

| season | n GW | L3 | L4 | Δ | L3 hits | L4 hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 31 | 1916 | 2052 | +136 | 72 | 80 |
| 2023-24 | 31 | 1825 | 1927 | +102 | 88 | 72 |
| 2024-25 | 31 | 2003 | 1955 | -48 | 104 | 60 |
| 2025-26 | 31 | 1842 | 1968 | +126 | 60 | 44 |

**Pooled:** L3 7586 · L4 7902 · Δ **+316**

## Verdict

Priced μ_post look-ahead **beats** L3 — promote as L4 default path.

JSON: `data/processed/layer4_mupost_priced.json`
