# Layer 3 climb sanity — raw μ vs μ_post transfers

Each arm wildcards GW0 on its μ, then rolls `solve_week_transfers`
(FT bank, max 2 hits). Score = realised XI + captain − hit points.

**Sanity only** — not a protocol gate. No chips / horizon.

## Season totals (net of hits)

| season | n GW | raw | shrink | Δ | raw hits | shrink hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 32 | 1948 | 1829 | -119 | 108 | 92 |
| 2023-24 | 33 | 2010 | 1834 | -176 | 108 | 92 |
| 2024-25 | 33 | 2061 | 2079 | +18 | 80 | 64 |
| 2025-26 | 33 | 2025 | 1987 | -38 | 60 | 48 |

**Pooled:** raw 8044 · shrink 7729 · Δ **-315**

## Paired GW Δpoints (shrink − raw)

- Mean Δ/GW: **-2.405** [-4.115, -0.763] (cluster bootstrap B=1000, seed=0)
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -3.719 |
| 2023-24 | -5.333 |
| 2024-25 | +0.545 |
| 2025-26 | -1.152 |

## Read

- Shrink **loses** on this single-week transfer climb: Δ **−2.4/GW**, interval below zero.
- Both arms take many hits (raw more than shrink) — expected with a **one-week** objective
  and no hold margin / horizon (Layer 4). That is optimiser's curse on path totals, not a
  rebuttal of the L2 forecast lock (`se_credibility`).
- Shrink is the more conservative transfer arm (fewer hits/transfers) but does not win
  realised net points here.
- **Do not** retune `HIT_COST` or drop μ_post from this table alone. Next lever is Layer 4
  (rolling horizon + terminal value / hold discipline).

JSON: `data/processed/layer3_climb.json`

