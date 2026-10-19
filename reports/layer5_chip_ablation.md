# Layer 5 chip ablation vs L4

**Confounded — do not use for chip ranking.** `no_chips` is the L5 **beam**, not L4’s horizon MILP; see `reports/layer5_nochips_diagnose.md`. Kept for the log only.

Diagnostic only — not a promote screen. Calendar boards depth=8, beam K=2. Each arm may play **only** the named chip (+ transfers); L4 = priced t:t+2 no chips.

Prior joint-climb dissection (`layer5_calendar_vs_l4`): chip **play** weeks were net **+114** vs L4; **none** weeks **−413**. Ablation asks which chip poisons the later path.

## Arm vs L4 (paired GW)

| arm | mean Δ/GW | lo | hi | covers 0 | pooled Δ | chip weeks | hits |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `no_chips` | -3.556 | -5.638 | -1.653 | False | -441 | 0 | 324 |
| `tc` | -2.911 | -5.008 | -0.863 | False | -361 | 8 | 324 |
| `bb` | -3.008 | -5.154 | -1.016 | False | -373 | 7 | 324 |
| `wc` | -5.629 | -7.953 | -3.225 | False | -698 | 8 | 368 |
| `fh` | -3.573 | -5.751 | -1.298 | False | -443 | 8 | 300 |

## By season (Δ pts)

| season | `no_chips` | `tc` | `bb` | `wc` | `fh` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | -122 | -108 | -111 | -257 | -91 |
| 2023-24 | -121 | -98 | -111 | -145 | -79 |
| 2024-25 | -26 | -13 | -5 | -71 | -118 |
| 2025-26 | -172 | -142 | -146 | -225 | -155 |

## Reading

- `no_chips` ≈ L5 transfer search on calendar boards with chips locked out — isolates board/horizon vs L4 priced path.
- A single-chip arm that clears `no_chips` helps; one that loses further is the poison.

JSON: `data/processed/layer5_chip_ablation.json`
