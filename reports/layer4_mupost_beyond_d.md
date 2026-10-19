# Layer 4 priced μ_post — beyond-t+2 arm D (team-form FDR)

Priced window **t:t+2** = expanding μ_post path (γ=1, unconstrained hits). Beyond t+2 never uses that week's line.

| arm | beyond t+2 |
| --- | --- |
| `A_hard_stop` | stop (H≈3) — current default |
| `D_team_form` | H=5; mean(priced μ)×tanh FDR scale; FDR = expanding club-pts μ_own−μ_opp (managers stripped) |

- H_extend = 5; γ = 1.0; max_hits = 15; terminal = 0
- FDR scale: 1 + 0.35·tanh(FDR/8.0), clipped to [0.55, 1.35]

## Primary: D − A

- Mean Δ/GW: **-2.266** [-4.024, -0.613]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -0.968 |
| 2023-24 | -1.323 |
| 2024-25 | -3.129 |
| 2025-26 | -3.645 |

## Vs L3

### A hard stop − L3

- Mean Δ/GW: **+3.556** [+1.653, +5.638]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +3.935 |
| 2023-24 | +3.903 |
| 2024-25 | +0.839 |
| 2025-26 | +5.548 |

### D team-form − L3

- Mean Δ/GW: **+1.290** [-0.976, +3.863]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +2.968 |
| 2023-24 | +2.581 |
| 2024-25 | -2.290 |
| 2025-26 | +1.903 |

## Season totals

| season | L3 | A | ΔA | D | ΔD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1916 | 2038 | +122 | 2008 | +92 |
| 2023-24 | 1825 | 1946 | +121 | 1905 | +80 |
| 2024-25 | 2003 | 2029 | +26 | 1932 | -71 |
| 2025-26 | 1842 | 2014 | +172 | 1901 | +59 |

**Pooled Δ vs L3:** A **+441** · D **+160**

## Verdict

D vs A **loses** to baseline.

A vs L3 **beats** baseline.

D vs L3 **inconclusive**.

Promote D only if D−A interval is clear of 0 above. Otherwise park long-horizon fill; keep A hard-stop.

JSON: `data/processed/layer4_mupost_beyond_d.json`
