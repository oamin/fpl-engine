# How FD λ scales relative to Opta ranking λ

Pre-registered **2026-10-29** · protocol `cascade_long_fd_opta_lam_scale_dig` · wall 0.0s

## Verdict

- **Not a single global scale.** Mean Opta−FD is **+0.068** in 2025-26 and **−0.078** in 2024-25.
- **Implementable form (venue mean-ratio):**

```
λ'_H = r_H · λ_H
λ'_A = r_A · λ_A
r_ha = mean(FD_ha) / mean(Opta_ha)
```

- **2025-26:** `r_H = 1.0638`, `r_A = 0.8473` (home boost, away shrink).
- **2024-25:** `r_H = 1.1663`, `r_A = 0.9390` (same direction, larger home boost).
- **Pooled 24+25 (candidate constants):** `r_H = 1.1146`, `r_A = 0.8928`.
- Direction is **stable** (FD warmer home / Opta hotter away); **magnitudes are not** — fitting on one season and applying to the other worsens venue bias. Do **not** promote without OOS + PI.
- Opta is also **more dispersed** than FD in 2025-26 (high Opta quintile F/O ≈ 0.83). Affine `FD ≈ a·Opta + b` (a < 1) compresses tails; pure scale cannot.

## Implement

1. After blind λ in `sample_match_from_opta_ranks` / `sample_opta_scenario` (same post-λ site as `away_shrink`).
2. Prefer explicit `home_scale` + existing `away_shrink`, or a pair `fd_venue_scale=(r_H, r_A)` — do not overload `LAM_SHRINK` (that scales both sides equally).
3. Defaults stay identity (`1.0`, `1.0`) until a pre-registered club-λ OOS screen clears the place×venue promote bar.

## 2025-26 fits

| lens | value |
| --- | ---: |
| n paired | 618 |
| mean Opta / FD | 1.517 / 1.449 |
| Opta−FD | +0.068 |
| corr | 0.706 |
| global FD/Opta | 0.9554 |
| OLS mul FD≈s·Opta | s=0.9146 |
| affine FD≈a·Opta+b | a=0.6042, b=+0.5326 |

| ha | FD/Opta | OLS s | affine a | affine b | Opta bias | FD bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H | 1.0638 | 1.0185 | 0.6673 | +0.6001 | -0.011 | +0.086 |
| A | 0.8473 | 0.8120 | 0.5471 | +0.4563 | +0.255 | +0.023 |

### Opta quintiles (compression)

| quintile | n | mean Opta | mean FD | FD/Opta | FD−Opta |
| --- | ---: | ---: | ---: | ---: | ---: |
| (0.516, 1.066] | 124 | 0.879 | 1.030 | 1.171 | +0.151 |
| (1.066, 1.249] | 123 | 1.155 | 1.250 | 1.082 | +0.095 |
| (1.249, 1.603] | 124 | 1.415 | 1.412 | 0.997 | -0.004 |
| (1.603, 1.97] | 123 | 1.765 | 1.596 | 0.904 | -0.169 |
| (1.97, 3.769] | 124 | 2.368 | 1.957 | 0.826 | -0.411 |

## Cross-season stability

| fit → apply | bias | bias H | bias A | MAE vs act | MAE vs FD |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw Opta on 2024-25 | +0.003 | -0.046 | +0.052 | 0.937 | 0.269 |
| r(2025) → 2024 | -0.064 | +0.049 | -0.176 | 0.938 | 0.241 |
| r(2024) → 2025 | +0.201 | +0.241 | +0.162 | 0.922 | 0.309 |
| pooled → 2025 | +0.127 | +0.162 | +0.092 | 0.907 | 0.284 |
| pooled → 2024 | +0.008 | +0.124 | -0.108 | 0.946 | 0.226 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_fd_vs_opta_lam_compare --season 2025-26
PYTHONPATH=. python3 -m src.eval.cascade_long_fd_vs_opta_lam_compare --season 2024-25 \
  --rows-out data/processed/cascade_long_fd_vs_opta_lam_compare_rows_2024-25.csv
PYTHONPATH=. python3 -m src.eval.cascade_long_fd_opta_lam_scale_dig
```

