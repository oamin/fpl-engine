# Layer 4 — beyond-t+2 D adaptive scales vs A

Priced **t:t+2** identical (γ=1, unconstrained hits). Fill weeks only:

| arm | beyond t+2 |
| --- | --- |
| `A_hard_stop` | stop (H≈3) |
| `D_adapt_const` | H=5; as-of OLS amp; scale = 1 + amp·tanh(FDR/8.0) |
| `D_adapt_decay` | H=5; as-of amp + ρ on grid; scale = 1 + amp·ρ^k·tanh(FDR/τ), k=0 at t+3 |

Fit: relative pts margin (÷ league team_μ) ~ amp · (ρ^k) · tanh(FDR/τ); pairs with target_gw < decision_gw only.

- H_extend = 5; γ = 1.0; max_hits = 15

## As-of fit summary (transfer weeks)

| arm | median amp | median ρ | median n | frac fallback |
| --- | ---: | ---: | ---: | ---: |
| `D_adapt_const` | 0.383 | 1.00 | 658 | 0.00 |
| `D_adapt_decay` | 0.402 | 0.95 | 658 | 0.00 |

## Primary: vs A

### D_adapt_const − A

- Mean Δ/GW: **-2.581** [-4.387, -0.837]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -1.677 |
| 2023-24 | -2.645 |
| 2024-25 | -2.032 |
| 2025-26 | -3.968 |

### D_adapt_decay − A

- Mean Δ/GW: **-1.855** [-3.598, -0.105]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -1.677 |
| 2023-24 | -1.355 |
| 2024-25 | -2.032 |
| 2025-26 | -2.355 |

## Vs L3

### A − L3

- Mean Δ/GW: **+3.556** [+1.653, +5.638]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +3.935 |
| 2023-24 | +3.903 |
| 2024-25 | +0.839 |
| 2025-26 | +5.548 |

### D_adapt_const − L3

- Mean Δ/GW: **+0.976** [-1.436, +3.597]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +2.258 |
| 2023-24 | +1.258 |
| 2024-25 | -1.194 |
| 2025-26 | +1.581 |

### D_adapt_decay − L3

- Mean Δ/GW: **+1.702** [-0.533, +4.291]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +2.258 |
| 2023-24 | +2.548 |
| 2024-25 | -1.194 |
| 2025-26 | +3.194 |

## Season totals

| season | L3 | A | ΔA | const | Δc | decay | Δd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1916 | 2038 | +122 | 1986 | +70 | 1986 | +70 |
| 2023-24 | 1825 | 1946 | +121 | 1864 | +39 | 1904 | +79 |
| 2024-25 | 2003 | 2029 | +26 | 1966 | -37 | 1966 | -37 |
| 2025-26 | 1842 | 2014 | +172 | 1891 | +49 | 1941 | +99 |

**Pooled Δ vs L3:** A **+441** · const **+121** · decay **+211**

## Verdict

const vs A **loses** to baseline.

decay vs A **loses** to baseline.

A vs L3 **beats** baseline.

Promote a D adapt arm only if it beats A with interval clear of 0 above.

JSON: `data/processed/layer4_mupost_beyond_d_adapt.json`
