Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L3 centreline H=12 climb vs H=2 priced — gated

Pre-registered: `protocol.l3_centreline_horizon_h12_climb`.

Strategiser screen on **locked** centreline μ (`protocol.l3_centreline_forecast`). Not a μ promote.

- Treatment `l3_centreline_h12`: `build_l3_centreline_horizon(H=12)` (FD/priced t:t+1 + Opta blind far).
- Baseline `l3_priced_h2`: `build_mupost_priced_horizon` (BOOKIE_DEPTH=2).
- γ=1.0; max_hits=15; shared μ_post wildcard @ GW8.
- Metric: paired GW realised XI+captain−hits (transfer weeks). Wall 645s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Result (H=12 − H=2)

- Mean Δ pts/GW: **-0.558** [-3.059, +2.034]
- Covers 0: **True**
- Verdict: **inconclusive**
- Pooled transfer-week Δ: **-67** (H12 7399 − H2 7466)

| season | n GW | H=2 | H=12 | Δ | H2 hits | H12 hits | H2 T | H12 T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1911 | 1925 | +14 | 168 | 84 | 73 | 52 | +0.467 |
| 2023-24 | 30 | 1839 | 1888 | +49 | 132 | 68 | 64 | 48 | +1.633 |
| 2024-25 | 30 | 1890 | 1831 | -59 | 108 | 100 | 58 | 56 | -1.967 |
| 2025-26 | 30 | 1826 | 1755 | -71 | 68 | 84 | 48 | 52 | -2.367 |

## Notes

- Positive Δ = H=12 centreline climb beats H=2 priced-only.
- Opta far uses season-final place ranks (same as locked horizon dig).
- Season totals are descriptive; the gate is the paired GW interval.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l3_centreline_h12_climb_gate
```

JSON: `data/processed/l3_centreline_h12_climb_gate.json`
