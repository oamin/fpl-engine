# Layer 4 — beyond E_decay (strength×γ + minutes decay) vs A

Priced window **t:t+1** (BOOKIE_DEPTH) = μ_post path (objective γ=1). Projected boards use minutes decay ρ=0.7 on last-3. Fill weeks: strength pots → score_xp_v2 × γ_fill^{k+1} (γ_fill=0.9). No book lines on fills.

| arm | policy |
| --- | --- |
| `A_hard_stop` | stop; expanding minutes — current default |
| `E_strength_gamma_decay` | H_extend; strength×γ_fill + minutes ρ=0.7 |

- H_extend = 5; objective γ = 1.0; γ_fill = 0.9; max_hits = 15

## Primary: E_decay − A

- Mean Δ/GW: **-1.919** [-4.049, +0.404]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -0.871 |
| 2023-24 | -4.581 |
| 2024-25 | -1.032 |
| 2025-26 | -1.194 |

## Vs L3

### A hard stop − L3

- Mean Δ/GW: **+1.556** [-0.589, +3.637]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +0.419 |
| 2023-24 | +2.581 |
| 2024-25 | +0.516 |
| 2025-26 | +2.710 |

### E_decay − L3

- Mean Δ/GW: **-0.363** [-2.823, +2.033]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -0.452 |
| 2023-24 | -2.000 |
| 2024-25 | -0.516 |
| 2025-26 | +1.516 |

## Season totals

| season | L3 | A | ΔA | E_decay | ΔE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1968 | 1981 | +13 | 1954 | -14 |
| 2023-24 | 1834 | 1914 | +80 | 1772 | -62 |
| 2024-25 | 1960 | 1976 | +16 | 1944 | -16 |
| 2025-26 | 1849 | 1933 | +84 | 1896 | +47 |

**Pooled Δ vs L3:** A **+193** · E_decay **-45**

## Verdict

E_decay vs A **inconclusive**.

A vs L3 **inconclusive**.

E_decay vs L3 **inconclusive**.

Promote E_decay only if E_decay−A interval lo > 0. Otherwise park; keep A hard-stop.

JSON: `data/processed/layer4_mupost_beyond_e_decay.json`
