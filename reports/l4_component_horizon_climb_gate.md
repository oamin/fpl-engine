Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4 component-μ H=12 vs H=2 climb — gated

Pre-registered: `protocol.l4_component_horizon_climb`.

Per-component μ screen on the locked L4 near / Opta far centreline recipes. Isolates which xP pieces help or hurt long-horizon transfer decisions. **Not** a μ promote.

- Treatment: `build_l3_centreline_horizon(H=12)` with `mu_post :=` component.
- Baseline: `build_mupost_priced_horizon` (BOOKIE_DEPTH=2) with `mu_post :=` same component.
- Near scorer: **L4** (`score_xp_v2` / priced FD); far: Opta blind λ → `score_team_side`.
- Shared wildcard open on full `score_xp_v2` @ GW8.
- No captaincy (objective + realised XI only); no chips; hits uncapped (max_hits=15, −4/hit); γ=1.0.
- Metric: paired GW realised XI−hits (transfer weeks). Wall 3196s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results (H=12 − H=2, same component μ)

### `xp_goals`

- Mean Δ pts/GW: **+2.692** [+0.583, +4.734]
- Covers 0: **False**
- Verdict: **promote**

| season | n GW | H=2 | H=12 | Δ | H2 hits | H12 hits | H2 T | H12 T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1666 | 1641 | -25 | 0 | 12 | 31 | 34 | -0.833 |
| 2023-24 | 30 | 1435 | 1527 | +92 | 0 | 12 | 31 | 34 | +3.067 |
| 2024-25 | 30 | 1373 | 1566 | +193 | 0 | 12 | 31 | 34 | +6.433 |
| 2025-26 | 30 | 1327 | 1390 | +63 | 0 | 20 | 31 | 36 | +2.100 |

### `xp_assists`

- Mean Δ pts/GW: **+2.292** [+0.575, +4.117]
- Covers 0: **False**
- Verdict: **promote**

| season | n GW | H=2 | H=12 | Δ | H2 hits | H12 hits | H2 T | H12 T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1653 | 1793 | +140 | 0 | 4 | 31 | 32 | +4.667 |
| 2023-24 | 30 | 1464 | 1570 | +106 | 0 | 4 | 31 | 32 | +3.533 |
| 2024-25 | 30 | 1394 | 1437 | +43 | 0 | 8 | 31 | 33 | +1.433 |
| 2025-26 | 30 | 1482 | 1468 | -14 | 0 | 16 | 31 | 35 | -0.467 |

### `xp_cs`

- Mean Δ pts/GW: **+0.267** [-2.275, +2.758]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | H=2 | H=12 | Δ | H2 hits | H12 hits | H2 T | H12 T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1260 | 1317 | +57 | 0 | 12 | 31 | 34 | +1.900 |
| 2023-24 | 30 | 1317 | 1240 | -77 | 0 | 16 | 31 | 35 | -2.567 |
| 2024-25 | 30 | 1188 | 1009 | -179 | 0 | 20 | 31 | 36 | -5.967 |
| 2025-26 | 30 | 1141 | 1372 | +231 | 0 | 16 | 31 | 35 | +7.700 |

### `xp_gc`

- Mean Δ pts/GW: **-0.858** [-2.359, +0.758]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | H=2 | H=12 | Δ | H2 hits | H12 hits | H2 T | H12 T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 730 | 609 | -121 | 0 | 4 | 31 | 32 | -4.033 |
| 2023-24 | 30 | 595 | 517 | -78 | 0 | 8 | 31 | 33 | -2.600 |
| 2024-25 | 30 | 601 | 637 | +36 | 0 | 0 | 31 | 31 | +1.200 |
| 2025-26 | 30 | 564 | 624 | +60 | 0 | 0 | 31 | 31 | +2.000 |

## Notes

- Positive Δ = H=12 (Opta far) beats H=2 priced-only when both climb on that component alone.
- Prior `l3_component_climb_gate` (component vs full, L3 one-week) answered a different question and stays parked.
- Season totals are descriptive; the gate is the paired GW interval.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_component_horizon_climb_gate
```

JSON: `data/processed/l4_component_horizon_climb_gate.json`
