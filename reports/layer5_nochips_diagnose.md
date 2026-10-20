# L5 no_chips first-order diagnose

Ablation stopped: ``no_chips`` was L5 **beam** (not L4 horizon MILP). This 2×2 separates **search** (horizon MILP vs beam) from **boards** (priced H≤3 vs calendar depth=8).

Baseline: **`l4_priced`**. Hits unconstrained; γ=1; shared μ-WC open.

## Season totals

| season | `l4_priced` | `l4_calendar` | `beam_priced` | `beam_calendar` |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | 2038 | 1984 | 1916 | 1916 |
| 2023-24 | 1946 | 1914 | 1825 | 1825 |
| 2024-25 | 2029 | 1984 | 2003 | 2003 |
| 2025-26 | 2014 | 1969 | 1842 | 1842 |

## Δ vs `l4_priced` (paired GW)

| arm | mean Δ/GW | lo | hi | covers 0 | pooled Δ |
| --- | ---: | ---: | ---: | --- | ---: |
| `l4_calendar` | -1.419 | -3.315 | +0.581 | True | -176 |
| `beam_priced` | -3.556 | -5.638 | -1.653 | False | -441 |
| `beam_calendar` | -3.556 | -5.638 | -1.653 | False | -441 |

## Reading

- **`beam_priced` −441** (lo < 0) → the L5 beam with chips locked is **myopic weekly transfers**, not L4’s horizon MILP. Chip ablation was fully confounded by this.
- **`beam_calendar` ≡ `beam_priced`** (same season totals) → with only `none`, depth-0 expansion has a single transfer child, so longer boards do not change the commit. Beam≠L4 is pure search.
- **`l4_calendar` −176** (covers 0) → longer boards under the **same** L4 policy are mild/inconclusive; not the first-order hole.
- **Next:** fix L5 transfer search to use horizon MILP (or equivalent foresight) before re-opening chips.

JSON: `data/processed/layer5_nochips_diagnose.json`
