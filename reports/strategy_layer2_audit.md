# Strategy Layer 2 — audit (μ_post shrinkage)

Ownership and η·SE **penalties are scrapped**. Layer 2 shrinks μ → μ_post,
then runs the same joint MILP on μ_post.

**Scope:** solver legality / objective identity on closed μ boards (as-of
moments, GW < t). Not a forecast proof — that is `reports/se_credibility.md`
(paired GW log-score vs realised points).

Mode: `constant`. Cases: **6/6 passed**.

| season | gw | L1 μ | L2 abs-μ | Δμ | L2 post-obj | overlap | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2022-23 | 10 | 83.92 | 83.49 | -0.43 | 69.76 | 9/11 | PASS |
| 2023-24 | 15 | 80.08 | 80.08 | +0.00 | 68.29 | 11/11 | PASS |
| 2024-25 | 20 | 85.88 | 85.88 | +0.00 | 73.64 | 11/11 | PASS |
| 2025-26 | 10 | 81.11 | 80.31 | -0.81 | 69.19 | 7/11 | PASS |
| 2025-26 | 25 | 75.88 | 75.52 | -0.36 | 66.29 | 8/11 | PASS |
| 2025-26 | 30 | 75.19 | 74.94 | -0.25 | 66.41 | 10/11 | PASS |

Δμ is absolute-μ of the L2 XI vs L1 (shrink can change the XI). Overlap is XI
player-id intersection. Forecast size/uncertainty: `reports/se_credibility.md`.

JSON: `data/processed/strategy_layer2_audit.json`

