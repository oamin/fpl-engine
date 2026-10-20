# Layer 4 μ_post ± SE(μ_post) — quick trial vs L3

Priced μ_post path (t:t+2). Decision: **`mu_dec = mu_post ± se_post`** with `se_post = K·xp_se`. No fixture-covariance draws, no within-position z.

- γ = 0.9; terminal = 0

## Paired GW Δpoints

### L4 +SE − L3

- Mean Δ/GW: **+0.879** [-1.363, +2.968]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -0.032 |
| 2023-24 | +2.065 |
| 2024-25 | -2.000 |
| 2025-26 | +3.484 |

### L4 −SE − L3

- Mean Δ/GW: **+2.048** [-0.161, +4.444]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +3.710 |
| 2023-24 | +3.516 |
| 2024-25 | -1.839 |
| 2025-26 | +2.806 |

## Season totals

| season | L3 | +SE | Δ+ | −SE | Δ− |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1899 | 1898 | -1 | 2014 | +115 |
| 2023-24 | 1813 | 1877 | +64 | 1922 | +109 |
| 2024-25 | 1996 | 1934 | -62 | 1939 | -57 |
| 2025-26 | 1840 | 1948 | +108 | 1927 | +87 |

**Pooled:** L3 7548 · +SE 7657 (Δ **+109**) · −SE 7802 (Δ **+254**)

## Verdict

+SE **inconclusive** vs L3.

−SE **inconclusive** vs L3.

Note: `se_post` is estimation SE of the shrunk mean (player-level), not player–player fixture covariance. Compare to mean-only priced L4 (+2.90/GW) and z+cov risk (+1.48/GW, covers 0).

JSON: `data/processed/layer4_mupost_se_pm.json`
