# Layer 4 adaptive posture — λ from L3−L4 cumulative gap

Lockstep climb. At GW **t**, adaptive L4 sees **cumulative points through t−1** only (`gap = L3_cum − L4_cum`). `λ_chase = λ_max·tanh(max(gap,0)/scale)`, `λ_cons = λ_max·tanh(max(−gap,0)/scale)`. `mu_dec = mu_post + λ_chase·z(up) − λ_cons·z(down)` from cov semi-SDs. No same-week realised leakage.

- λ_max = **0.35**; gap_scale = **30.0** pts
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

- Mean Δ/GW: **+0.968** [-1.387, +3.218]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +0.742 |
| 2023-24 | +1.387 |
| 2024-25 | -0.935 |
| 2025-26 | +2.677 |

### Adaptive − expanding

- Mean Δ/GW: **-1.927** [-3.581, -0.145]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -4.000 |
| 2023-24 | -2.613 |
| 2024-25 | +0.355 |
| 2025-26 | -1.452 |

## Season totals + posture mix

| season | L3 | exp | adapt | Δadapt | chase% | cons% | neut% | mean |gap| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1899 | 2046 | 1922 | +23 | 27% | 63% | 10% | 16.7 |
| 2023-24 | 1813 | 1937 | 1856 | +43 | 17% | 77% | 7% | 18.2 |
| 2024-25 | 1996 | 1956 | 1967 | -29 | 67% | 20% | 13% | 19.7 |
| 2025-26 | 1840 | 1968 | 1923 | +83 | 3% | 93% | 3% | 32.0 |

**Pooled:** L3 7548 · exp 7907 (Δ **+359**) · adapt 7668 (Δ **+120**) · adapt−exp **-239**

## Verdict

Expanding vs L3 **beats** baseline.

Adaptive vs L3 **inconclusive**.

Adaptive vs expanding **loses** to baseline.

JSON: `data/processed/layer4_mupost_posture.json`
