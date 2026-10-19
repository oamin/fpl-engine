# Layer 4 chase — upside semi-SD vs expanding / L3

Priced μ_post path (t:t+2). **Expanding** = mean-only `mu_post`. **Chase** = `mu_dec = mu_post + λ · z(upside_semi_sd)` from fixture-cov draws (RSS over DGW). Always-on posture stress test. No SE blend; μ_post intact.

- λ = **0.35**; n_scenarios = 32
- γ = 0.9; terminal = 0

## Vs L3

### Expanding − L3

- Mean Δ/GW: **+2.895** [+0.677, +5.194]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +4.742 |
| 2023-24 | +4.000 |
| 2024-25 | -1.290 |
| 2025-26 | +4.129 |

### Chase − L3

- Mean Δ/GW: **+2.339** [+0.338, +4.419]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +3.323 |
| 2023-24 | +3.968 |
| 2024-25 | -1.129 |
| 2025-26 | +3.194 |

## Chase − expanding

- Mean Δ/GW: **-0.556** [-2.323, +1.218]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -1.419 |
| 2023-24 | -0.032 |
| 2024-25 | +0.161 |
| 2025-26 | -0.935 |

## Season totals

| season | L3 | exp | Δexp | chase | Δchase | chase−exp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1899 | 2046 | +147 | 2002 | +103 | -44 |
| 2023-24 | 1813 | 1937 | +124 | 1936 | +123 | -1 |
| 2024-25 | 1996 | 1956 | -40 | 1961 | -35 | +5 |
| 2025-26 | 1840 | 1968 | +128 | 1939 | +99 | -29 |

**Pooled:** L3 7548 · exp 7907 (Δ **+359**) · chase 7838 (Δ **+290**) · chase−exp **-69**

## Verdict

Expanding vs L3 **beats** baseline.

Chase vs L3 **beats** baseline.

Chase vs expanding **inconclusive**.

JSON: `data/processed/layer4_mupost_chase.json`
