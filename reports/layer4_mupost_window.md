# Layer 4 expanding vs windowed μ_post

Priced path t:t+2. **Expanding** = locked L4 priced (week-t = L3 `mu_post`). **Window W** = rates/priors/minutes + cohort moments on last W GWs; project t:t+2 then same constant-K shrink (week-t also windowed projection).

- γ = 0.9; terminal = 0; windows = [3, 5]

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

### W=3 − L3

- Mean Δ/GW: **-10.774** [-13.823, -7.492]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -10.032 |
| 2023-24 | -7.581 |
| 2024-25 | -11.677 |
| 2025-26 | -13.806 |

### W=5 − L3

- Mean Δ/GW: **-7.105** [-10.041, -3.815]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -6.935 |
| 2023-24 | -3.226 |
| 2024-25 | -6.452 |
| 2025-26 | -11.806 |

## Vs expanding L4

### W=3 − expanding

- Mean Δ/GW: **-13.669** [-16.896, -10.242]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -14.774 |
| 2023-24 | -11.581 |
| 2024-25 | -10.387 |
| 2025-26 | -17.935 |

### W=5 − expanding

- Mean Δ/GW: **-10.000** [-13.017, -6.830]
- Covers zero: **False**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -11.677 |
| 2023-24 | -7.226 |
| 2024-25 | -5.161 |
| 2025-26 | -15.935 |

## Season totals

| season | L3 | exp | Δexp | W3 | ΔW3 | W5 | ΔW5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1899 | 2046 | +147 | 1588 | -311 | 1684 | -215 |
| 2023-24 | 1813 | 1937 | +124 | 1578 | -235 | 1713 | -100 |
| 2024-25 | 1996 | 1956 | -40 | 1634 | -362 | 1796 | -200 |
| 2025-26 | 1840 | 1968 | +128 | 1412 | -428 | 1474 | -366 |

**Pooled Δ vs L3:** exp **+359** · W3 **-1336** · W5 **-881**

## Verdict

Expanding vs L3 **beats** baseline.

W=3 vs L3 **loses** to baseline.

W=5 vs L3 **loses** to baseline.

W=3 vs expanding **loses** to baseline.

W=5 vs expanding **loses** to baseline.

JSON: `data/processed/layer4_mupost_window.json`
