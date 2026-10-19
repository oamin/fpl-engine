# Weekly MILP climb — raw μ vs constant μ_post

Each GW: joint MILP rebuild of 15+XI+captain (no transfer link). Choose on
`score_xp_v2` (raw) or constant-K μ_post; score = realised XI + captain points.

**Not** the locked forecast gate (`se_credibility`). Not a free-transfer climb
(Layer 3). Sanity check only.

Gameweeks: 131 across four closed seasons (from GW6).
Same XI+captain: **58.8%**; mean XI overlap **10.4/11**.

## Season totals

| season | n GW | raw pts | shrink pts | Δ | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 32 | 2312 | 2294 | -18 | -0.56 |
| 2023-24 | 33 | 2201 | 2181 | -20 | -0.61 |
| 2024-25 | 33 | 2337 | 2324 | -13 | -0.39 |
| 2025-26 | 33 | 2296 | 2250 | -46 | -1.39 |

**Pooled:** raw 9146 · shrink 9049 · Δ **-97**

## Paired GW Δpoints (shrink − raw)

- Mean Δ/GW: **-0.740** [-1.657, +0.206] (cluster bootstrap B=1000, seed=0)
- Covers zero: **True**

## Read

- Positive Δ ⇒ constant shrink gained realised points on the weekly rebuild.
- A small / zero-covering interval is expected: within-position order is mostly
  unchanged under constant K, so XIs often match; the forecast win is on
  player-GW log-score, not necessarily on optimiser totals.

JSON: `data/processed/shrink_climb.json`

