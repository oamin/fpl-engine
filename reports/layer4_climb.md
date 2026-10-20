# Layer 4 climb sanity — horizon vs Layer 3 (μ_post)

Both arms wildcard on μ_post, then L3 one-week vs L4 H=4 (γ=0.9, terminal_μ=0.25, path=repeat).
Score = realised XI + captain − hits. **Sanity only** — not a gate.

- Path **repeat**: future boards = decision-week μ (as-of).
- Path **oracle**: future closed μ boards (optimistic upper bound).

| season | n GW | L3 | L4 | Δ | L3 hits | L4 hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 32 | 1829 | 1917 | +88 | 92 | 220 |
| 2023-24 | 33 | 1834 | 1958 | +124 | 92 | 228 |
| 2024-25 | 33 | 2079 | 1998 | -81 | 64 | 220 |
| 2025-26 | 33 | 1987 | 1933 | -54 | 48 | 188 |

**Pooled:** L3 7729 · L4 7806 · Δ **+77**

## Paired GW Δpoints (L4 − L3)

- Mean Δ/GW: **+0.588** [-1.107, +2.435]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +2.750 |
| 2023-24 | +3.758 |
| 2024-25 | -2.455 |
| 2025-26 | -1.636 |

Does **not** overturn L3; informs whether horizon helps before Layer 5.

JSON: `data/processed/layer4_climb.json`
