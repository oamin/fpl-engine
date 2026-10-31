Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L3 component climb — gated (no captain, uncapped hits)

Pre-registered: `protocol.l3_component_climb_gate`.

Same as-of FD centreline board each GW. Arms differ only by μ column.
Baseline `full` = `score_xp_v2`. Treatments: `xp_goals`, `xp_assists`, `xp_cs`, `xp_gc`.

- **No captaincy** in objective or realised score (XI points only).
- Hits uncapped (solver bound max_hits=15; still −4/hit).
- Shared wildcard open on full `score_xp_v2` @ GW8.
- Wall 154s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## xp_goals − full

- Mean Δ pts/GW: **-7.412** [-10.757, -3.898]
- Covers 0: **False**
- Verdict: **park_clear_lose**

| season | n GW | full | component | Δ | full hits | comp hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1845 | 1660 | -185 | 60 | 0 | -6.167 |
| 2023-24 | 29 | 1534 | 1315 | -219 | 24 | 0 | -7.552 |
| 2024-25 | 30 | 1561 | 1290 | -271 | 28 | 0 | -9.033 |
| 2025-26 | 30 | 1484 | 1277 | -207 | 24 | 0 | -6.900 |

## xp_assists − full

- Mean Δ pts/GW: **-6.773** [-9.937, -3.604]
- Covers 0: **False**
- Verdict: **park_clear_lose**

| season | n GW | full | component | Δ | full hits | comp hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1845 | 1774 | -71 | 60 | 0 | -2.367 |
| 2023-24 | 29 | 1534 | 1383 | -151 | 24 | 0 | -5.207 |
| 2024-25 | 30 | 1561 | 1187 | -374 | 28 | 0 | -12.467 |
| 2025-26 | 30 | 1484 | 1274 | -210 | 24 | 0 | -7.000 |

## xp_cs − full

- Mean Δ pts/GW: **-15.151** [-18.530, -11.865]
- Covers 0: **False**
- Verdict: **park_clear_lose**

| season | n GW | full | component | Δ | full hits | comp hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1845 | 1448 | -397 | 60 | 0 | -13.233 |
| 2023-24 | 29 | 1534 | 986 | -548 | 24 | 0 | -18.897 |
| 2024-25 | 30 | 1561 | 1178 | -383 | 28 | 0 | -12.767 |
| 2025-26 | 30 | 1484 | 1009 | -475 | 24 | 0 | -15.833 |

## xp_gc − full

- Mean Δ pts/GW: **-26.328** [-31.650, -21.504]
- Covers 0: **False**
- Verdict: **park_clear_lose**

| season | n GW | full | component | Δ | full hits | comp hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1845 | 1130 | -715 | 60 | 0 | -23.833 |
| 2023-24 | 29 | 1534 | 879 | -655 | 24 | 0 | -22.586 |
| 2024-25 | 30 | 1561 | 655 | -906 | 28 | 0 | -30.200 |
| 2025-26 | 30 | 1484 | 627 | -857 | 24 | 0 | -28.567 |

## Notes

- Positive Δ = component-μ climb beats full `score_xp_v2` climb.
- Diagnostic isolation of which centreline piece drives transfer value.
- Season totals descriptive; gate = paired GW interval per component.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l3_component_climb_gate
```

JSON: `data/processed/l3_component_climb_gate.json`
