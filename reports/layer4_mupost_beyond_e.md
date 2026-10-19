# Layer 4 — beyond arm E (strength pots × γ^k) vs A

Priced window **t:t+1** (BOOKIE_DEPTH) = μ_post path (objective γ=1, unconstrained hits). Fill weeks: as-of team-strength xG/xCS/xGC → `score_xp_v2`, then μ *= γ_fill^{k+1} (γ_fill=0.9). No book lines on fills.

| arm | beyond priced window |
| --- | --- |
| `A_hard_stop` | stop — current default |
| `E_strength_gamma` | H_extend; strength pots × γ_fill^{k+1} |

- H_extend = 5; objective γ = 1.0; γ_fill = 0.9; max_hits = 15

## Primary: E − A

- Mean Δ/GW: **+0.097** [-1.871, +1.984]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +1.032 |
| 2023-24 | -2.516 |
| 2024-25 | +1.516 |
| 2025-26 | +0.355 |

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

### E strength×γ − L3

- Mean Δ/GW: **+1.653** [-0.766, +4.066]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +1.452 |
| 2023-24 | +0.065 |
| 2024-25 | +2.032 |
| 2025-26 | +3.065 |

## Season totals

| season | L3 | A | ΔA | E | ΔE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1968 | 1981 | +13 | 2013 | +45 |
| 2023-24 | 1834 | 1914 | +80 | 1836 | +2 |
| 2024-25 | 1960 | 1976 | +16 | 2023 | +63 |
| 2025-26 | 1849 | 1933 | +84 | 1944 | +95 |

**Pooled Δ vs L3:** A **+193** · E **+205**

## Verdict

E vs A **inconclusive**.

A vs L3 **inconclusive**.

E vs L3 **inconclusive**.

Promote E only if E−A interval lo > 0. Otherwise park; keep A hard-stop.

JSON: `data/processed/layer4_mupost_beyond_e.json`
