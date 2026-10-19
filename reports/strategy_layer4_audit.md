# Strategy Layer 4 — horizon audit

Transfers only at the decision week; XI+captain each week in H on that 15.
γ = 0.9, default H = 4, terminal μ weight = 0.25.

Cases passed: **6/6**

| season | gw | pass | H=1≡L3 | H transfers≠L3 | H obj | L3 obj |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 2022-23 | 10 | PASS | yes | 4 | 209.36 | 69.76 |
| 2023-24 | 15 | PASS | yes | 3 | 208.89 | 68.29 |
| 2024-25 | 20 | PASS | yes | 3 | 240.60 | 73.64 |
| 2025-26 | 10 | PASS | yes | 2 | 231.65 | 69.19 |
| 2025-26 | 25 | PASS | yes | 3 | 231.94 | 66.29 |
| 2025-26 | 30 | PASS | yes | 3 | 227.72 | 66.41 |

## Reading

- **H=1≡L3:** identity required (gates the wrapper).
- **H transfers≠L3:** count of symmetric-diff buys; look-ahead can differ.
- Climb sanity (realised points) is separate: `python -m src.strategy.layer4_climb`.

JSON: `data/processed/strategy_layer4_audit.json`
