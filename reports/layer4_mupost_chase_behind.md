# Layer 4 adaptive chase-only when behind

Lockstep climb. At GW **t**, adaptive L4 sees **cumulative points through t−1** only (`gap = L3_cum − L4_cum`). `λ_chase = λ_max·tanh(max(gap,0)/scale)`; **λ_cons = 0** (neutral when ahead). `mu_dec = mu_post + λ_chase·z(up) − λ_cons·z(down)` from cov semi-SDs. No same-week realised leakage.

- λ_max = **0.35**; gap_scale = **30.0** pts; **chase_only**
- n_scenarios = 32; γ = 0.9

## Paired GW Δpoints

### Expanding − L3

- Mean Δ/GW: **+2.895** [+0.677, +5.194]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +4.742 |
| 2023-24 | +4.000 |
| 2024-25 | -1.290 |
| 2025-26 | +4.129 |

### Adaptive − L3

- Mean Δ/GW: **+2.855** [+0.515, +5.355]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +4.226 |
| 2023-24 | +4.000 |
| 2024-25 | -0.935 |
| 2025-26 | +4.129 |

### Adaptive − expanding

- Mean Δ/GW: **-0.040** [-1.065, +0.960]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -0.516 |
| 2023-24 | +0.000 |
| 2024-25 | +0.355 |
| 2025-26 | +0.000 |

## Season totals + posture mix

| season | L3 | exp | adapt | Δadapt | chase% | cons% | neut% | mean |gap| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1899 | 2046 | 2030 | +131 | 10% | 0% | 90% | 43.5 |
| 2023-24 | 1813 | 1937 | 1937 | +124 | 0% | 0% | 100% | 42.9 |
| 2024-25 | 1996 | 1956 | 1967 | -29 | 67% | 0% | 33% | 19.7 |
| 2025-26 | 1840 | 1968 | 1968 | +128 | 7% | 0% | 93% | 44.1 |

**Pooled:** L3 7548 · exp 7907 (Δ **+359**) · adapt 7902 (Δ **+354**) · adapt−exp **-5**

## Verdict

Expanding vs L3 **beats** baseline.

Adaptive vs L3 **beats** baseline.

Adaptive vs expanding **inconclusive**.

JSON: `data/processed/layer4_mupost_chase_behind.json`
