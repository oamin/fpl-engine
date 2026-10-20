# Layer 4 μ_post priced path — GW(t:t+2) vs L3

Week **t** = Layer-3 `mu_post`. Weeks **t+1, t+2** = projected `score_xp_v2` (frozen priors + FD opening pots) then the same constant-K shrink. Hard stop (no beyond-t+2). Covariance draws not used. Hits unconstrained (solver bound max_hits=15); objective still −4 per hit.

- γ = 1.0; terminal = 0; max_hits = 15

## Paired GW Δpoints (L4 − L3)

- Mean Δ/GW: **+3.556** [+1.653, +5.638]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +3.935 |
| 2023-24 | +3.903 |
| 2024-25 | +0.839 |
| 2025-26 | +5.548 |

## Season totals

| season | n GW | L3 | L4 | Δ | L3 hits | L4 hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 31 | 1916 | 2038 | +122 | 72 | 80 |
| 2023-24 | 31 | 1825 | 1946 | +121 | 88 | 68 |
| 2024-25 | 31 | 2003 | 2029 | +26 | 104 | 92 |
| 2025-26 | 31 | 1842 | 2014 | +172 | 60 | 44 |

**Pooled:** L3 7586 · L4 8027 · Δ **+441**

## Verdict

Priced μ_post with γ=1 **beats** L3 — **L4 default path** (undiscounted; prior γ=0.9 in `reports/layer4_mupost_priced.md`).

JSON: `data/processed/layer4_mupost_gamma1.json`
