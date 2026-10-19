# Layer 4 priced μ_post — beyond-t+2 arms A/B/C

Priced window **t:t+2** = expanding μ_post path. Beyond t+2 never uses that week's line.

| arm | beyond t+2 |
| --- | --- |
| `A_hard_stop` | stop (H≈3) — current default |
| `B_crude_fdr` | H=5; mean(priced μ)×as-of FDR vs opponent |
| `C_decay` | H=5; γ_fill=0.9 blend last priced → pos mean |

- H_extend = 5; γ = 0.9; terminal = 0

## Vs L3

### A hard stop − L3

- Mean Δ/GW: **+2.895** [+0.677, +5.194]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +4.742 |
| 2023-24 | +4.000 |
| 2024-25 | -1.290 |
| 2025-26 | +4.129 |

### B crude FDR − L3

- Mean Δ/GW: **+2.508** [+0.266, +4.839]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +2.097 |
| 2023-24 | +3.226 |
| 2024-25 | +0.516 |
| 2025-26 | +4.194 |

### C decay − L3

- Mean Δ/GW: **+1.782** [-0.218, +3.912]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +1.355 |
| 2023-24 | +2.258 |
| 2024-25 | -0.645 |
| 2025-26 | +4.161 |

## Vs A (hard stop)

### B − A

- Mean Δ/GW: **-0.387** [-1.944, +1.202]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -2.645 |
| 2023-24 | -0.774 |
| 2024-25 | +1.806 |
| 2025-26 | +0.065 |

### C − A

- Mean Δ/GW: **-1.113** [-2.758, +0.540]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -3.387 |
| 2023-24 | -1.742 |
| 2024-25 | +0.645 |
| 2025-26 | +0.032 |

## Season totals

| season | L3 | A | ΔA | B | ΔB | C | ΔC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1899 | 2046 | +147 | 1964 | +65 | 1941 | +42 |
| 2023-24 | 1813 | 1937 | +124 | 1913 | +100 | 1883 | +70 |
| 2024-25 | 1996 | 1956 | -40 | 2012 | +16 | 1976 | -20 |
| 2025-26 | 1840 | 1968 | +128 | 1970 | +130 | 1969 | +129 |

**Pooled Δ vs L3:** A **+359** · B **+311** · C **+221**

## Verdict

A vs L3 **beats** baseline.

B vs L3 **beats** baseline.

C vs L3 **inconclusive**.

B vs A **inconclusive**.

C vs A **inconclusive**.

JSON: `data/processed/layer4_mupost_beyond.json`
