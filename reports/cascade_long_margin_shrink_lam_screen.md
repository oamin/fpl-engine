# cascade_long Opta margin-shrink λ screen

Pre-registered **2026-10-29** · protocol `cascade_long_margin_shrink_lam_screen` · wall 13.9s

## Lever

Strength-mismatch from Opta−FD residual drivers (not form, not venue spice). Compress fixture Opta margins toward the match mean:

```
m = (λ_H + λ_A) / 2
λ'_H = m + α · (λ_H − m)
λ'_A = m + α · (λ_A − m)
```

Train-fit α (MAE on ('2022-23', '2023-24', '2024-25')): **0.90**.

## Holdout 2025-26

| arm | α | bias | bias H | bias A | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_blind | 1.00 | +0.122 | -0.015 | +0.259 | 0.918 |
| margin_shrink_fit | 0.90 | +0.122 | -0.014 | +0.259 | 0.914 |
| margin_shrink_075 | 0.75 | +0.122 | -0.013 | +0.258 | 0.913 |

## Promote vs baseline

- **margin_shrink_fit**: FAIL (away_improve=+0.000, home_worsen=-0.000, mae_worsen=-0.004)
- **margin_shrink_075**: FAIL (away_improve=+0.001, home_worsen=-0.001, mae_worsen=-0.006)

## OOS

- **2022-23**: winners=['none']
- **2023-24**: winners=['none']
- **2024-25**: winners=['none']
- **2025-26**: winners=['none']

### OOS pass counts

- margin_shrink_fit: 0/4
- margin_shrink_075: 0/4

**Recommended:** none — keep blind α=1

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_margin_shrink_lam_screen
```

