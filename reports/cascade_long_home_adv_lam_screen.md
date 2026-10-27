# cascade_long HOME_ADV club-λ venue screen

Pre-registered **2026-10-27** · protocol `cascade_long_home_adv_lam_screen` · wall 203.9s

## Locked before holdout

- Baseline: `home_adv=1.0` (PI lock).
- Candidate: `home_adv=1.08`.
- Fit: argmin train MAE on 2022-23, 2023-24, 2024-25 → **`home_adv=1.06`**.
- Holdout: **2025-26** GW8–38.
- Promote if vs baseline: away |bias| ↓ ≥0.1, home |bias| ↑ ≤0.05, MAE ↑ ≤0.02.
- Default `HOME_ADV` **not** flipped here (PI re-approval required).

## Holdout 2025-26 (Opta λ − actual)

| arm | home_adv | bias | bias H | bias A | MAE | FD bias A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_1_00 | 1.00 | +0.124 | -0.007 | +0.255 | 0.917 | +0.023 |
| candidate_1_08 | 1.08 | +0.128 | +0.112 | +0.143 | 0.908 | +0.023 |
| fit_train | 1.06 | +0.126 | +0.082 | +0.170 | 0.909 | +0.023 |

## Promote decisions vs baseline

- **candidate_1_08**: FAIL (away_improve=+0.112, home_worsen=+0.105, mae_worsen=-0.009)
- **fit_train**: FAIL (away_improve=+0.085, home_worsen=+0.075, mae_worsen=-0.008)

**Recommended (counterfactual only):** none — keep HOME_ADV=1.0

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_home_adv_lam_screen
```

