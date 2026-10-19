# Layer 4 bookie-path trial — GW(t:t+2) + arms A/B/C

Priors frozen at GW `< t`. Bookie window = **t:t+2** (FD opening AH+OU → fixture-covariance draw-mean μ). Beyond t+2 does **not** use that week's line.

- Scenarios / fixture: **32**
- H_extend (B/C): **5**
- γ_fill (C): **0.9**
- Terminal weights: **0** (path contrast only)

## Arms

| arm | beyond t+2 |
| --- | --- |
| `A_hard_stop` | stop (H≈3) |
| `B_crude_fdr` | extend; mean(priced μ)×as-of FDR vs opponent |
| `C_decay` | extend; decay last priced μ → position mean |

## Paired GW Δpoints (arm − L3)

| contrast | mean | lo | hi | covers 0? |
| --- | ---: | ---: | ---: | --- |
| `A_hard_stop_minus_l3` | -5.556 | -8.347 | -2.871 | no |
| `B_crude_fdr_minus_l3` | -8.000 | -10.653 | -5.322 | no |
| `C_decay_minus_l3` | -7.653 | -10.501 | -4.685 | no |

## Season totals

| season | arm | n GW | L3 | arm | Δ | L3 hits | arm hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | A_hard_stop | 31 | 1899 | 1724 | -175 | 72 | 88 |
| 2023-24 | A_hard_stop | 31 | 1813 | 1868 | +55 | 80 | 56 |
| 2024-25 | A_hard_stop | 31 | 1996 | 1655 | -341 | 76 | 72 |
| 2025-26 | A_hard_stop | 31 | 1840 | 1612 | -228 | 52 | 40 |
| 2022-23 | B_crude_fdr | 31 | 1899 | 1658 | -241 | 72 | 140 |
| 2023-24 | B_crude_fdr | 31 | 1813 | 1694 | -119 | 80 | 128 |
| 2024-25 | B_crude_fdr | 31 | 1996 | 1622 | -374 | 76 | 144 |
| 2025-26 | B_crude_fdr | 31 | 1840 | 1582 | -258 | 52 | 92 |
| 2022-23 | C_decay | 31 | 1899 | 1592 | -307 | 72 | 132 |
| 2023-24 | C_decay | 31 | 1813 | 1814 | +1 | 80 | 88 |
| 2024-25 | C_decay | 31 | 1996 | 1639 | -357 | 76 | 124 |
| 2025-26 | C_decay | 31 | 1840 | 1554 | -286 | 52 | 108 |

## Verdict

A_hard_stop loses to L3; B_crude_fdr loses to L3; C_decay loses to L3.

All three beyond-t+2 policies lose to L3 on this climb. Hard stop is least bad. Keep L3 as the transfer default; bookie path stays a research arm (μ calibration / hit discipline / minutes prior next).

JSON: `data/processed/layer4_bookie_trial.json`
