# Layer 4 — team-form terminal vs A hard-stop

Both arms: priced **t:t+2** μ_post boards (H≈3). **No fill weeks.**

| arm | terminal |
| --- | --- |
| `A_hard_stop` | 0 |
| `T_form_terminal` | γ^H · credit_i; credit_i = amp · Σ_k tanh(FDR_{t+3+k}) · mean_priced_μ_i; amp as-of OLS |

- γ = 1.0; max_hits = 15; n_beyond = 2 (leads 3…4)

## As-of fit (T arm)

- median amp **0.383** · median n **658** · median mean\|credit\| **1.012** · fallback frac **0.00**

## Primary: T − A

- Mean Δ/GW: **-3.274** [-5.016, -1.516]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -2.645 |
| 2023-24 | -4.129 |
| 2024-25 | -1.516 |
| 2025-26 | -4.806 |

## Vs L3

### A − L3

- Mean Δ/GW: **+3.556** [+1.653, +5.638]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +3.935 |
| 2023-24 | +3.903 |
| 2024-25 | +0.839 |
| 2025-26 | +5.548 |

### T − L3

- Mean Δ/GW: **+0.282** [-1.808, +2.484]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +1.290 |
| 2023-24 | -0.226 |
| 2024-25 | -0.677 |
| 2025-26 | +0.742 |

## Season totals

| season | L3 | A | ΔA | T | ΔT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1916 | 2038 | +122 | 1956 | +40 |
| 2023-24 | 1825 | 1946 | +121 | 1818 | -7 |
| 2024-25 | 2003 | 2029 | +26 | 1982 | -21 |
| 2025-26 | 1842 | 2014 | +172 | 1865 | +23 |

**Pooled Δ vs L3:** A **+441** · T **+35**

## Verdict

T vs A **loses** to baseline.

A vs L3 **beats** baseline.

T vs L3 **inconclusive**.

Promote T only if T−A interval is clear of 0 above; else keep A.

JSON: `data/processed/layer4_mupost_terminal.json`
