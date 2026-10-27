# Combined λ shrink screen (λ_H and λ_A × s)

Pre-registered **2026-10-27** · wall 233.6s

## Formula

- `λ_H ← λ_H · s`, `λ_A ← λ_A · s` (`away_shrink=1.0`).
- No 0.20–3.50 λ clip.
- Default `LAM_SHRINK=1.0` **not** flipped here.

Train fit → **s=0.92**. Holdout **2025-26** GW8–38.
Promote: overall |bias| ↓ ≥0.05, side |bias| ↑ ≤0.05, MAE ↑ ≤0.02.

## Holdout 2025-26

| arm | s | bias | bias H | bias A | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_1_00 | 1.00 | +0.125 | -0.007 | +0.256 | 0.917 |
| fit_train | 0.92 | +0.003 | -0.128 | +0.135 | 0.897 |

## Holdout promote

- **fit_train**: FAIL (overall_improve=+0.121, home_worsen=+0.121, away_worsen=-0.121, mae_worsen=-0.020)

## OOS prior seasons at fit s=0.92

| season | s=1 bias | s=fit bias | bias H Δ|bias| | bias A Δ|bias| | promote |
| --- | ---: | ---: | ---: | ---: | --- |
| 2022-23 | +0.074 | -0.042 | +0.117 | -0.114 | FAIL |
| 2023-24 | -0.240 | -0.351 | +0.113 | +0.108 | FAIL |
| 2024-25 | +0.015 | -0.100 | +0.116 | -0.025 | FAIL |

**OOS PASS count:** 0/3

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_lam_shrink_screen
```

