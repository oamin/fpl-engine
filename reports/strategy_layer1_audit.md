# Strategy Layer 1 — independent audit

Scope: single-week joint MILP on absolute `score_xp_v2` μ only.
No ownership, SE, transfers, chips, or horizon.

Cases: **6/6 passed** all checks.

## Checks per GW

| season | gw | pool | joint obj | seq obj | gap | cost | form | captain | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2022-23 | 10 | 212 | 83.92 | 83.68 | +0.246 | 986 | 1-3-5-2 | Erling Haaland | PASS |
| 2023-24 | 15 | 208 | 80.08 | 80.08 | -0.000 | 978 | 1-3-5-2 | Mohamed Salah | PASS |
| 2024-25 | 20 | 200 | 85.88 | 85.13 | +0.748 | 998 | 1-3-5-2 | Mohamed Salah | PASS |
| 2025-26 | 10 | 204 | 81.11 | 81.03 | +0.077 | 947 | 1-3-5-2 | Erling Haaland | PASS |
| 2025-26 | 25 | 198 | 75.88 | 75.55 | +0.328 | 995 | 1-3-5-2 | Gabriel dos Santos Magalhães | PASS |
| 2025-26 | 30 | 200 | 75.19 | 74.52 | +0.666 | 998 | 1-3-5-2 | Erling Haaland | PASS |

## Failed checks (if any)

None.

## Method

- Pool: `score_xp_v2_closed.csv` (μ summed over DGW fixtures) joined to `score_xp_*` values.
- Joint: `src.strategy.layer1_week.solve_week`.
- Sequential baseline: `pick_squad` then `pick_xi`; captain = max μ in XI.
- Expectation: joint objective ≥ sequential (same μ, joint formation+captain).

JSON: `data/processed/strategy_layer1_audit.json`

