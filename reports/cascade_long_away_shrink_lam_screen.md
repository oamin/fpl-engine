# cascade_long away_shrink club-λ venue screen

Pre-registered **2026-10-27** · protocol `cascade_long_away_shrink_lam_screen` · wall 206.2s

## Formula

- `λ_H` unchanged.
- `λ_A ← λ_A · away_shrink`.
- No hard 0.20–3.50 λ clip.
- Supersedes symmetric `home_adv` screen.

## Locked before holdout

- Baseline: `away_shrink=1.0` (default).
- Candidate: `away_shrink=0.93` (≈1/1.08).
- Fit: argmin train MAE on 2022-23, 2023-24, 2024-25 → **`away_shrink=0.86`**.
- Holdout: **2025-26** GW8–38.
- Promote if vs baseline: away |bias| ↓ ≥0.1, home |bias| ↑ ≤0.05, MAE ↑ ≤0.02.
- Default `AWAY_SHRINK` **not** flipped here (PI re-approval required).

## Holdout 2025-26 (Opta λ − actual)

| arm | away_shrink | bias | bias H | bias A | MAE | FD bias A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_1_00 | 1.00 | +0.125 | -0.007 | +0.256 | 0.917 | +0.023 |
| candidate_0_93 | 0.93 | +0.072 | -0.007 | +0.150 | 0.902 | +0.023 |
| fit_train | 0.86 | +0.018 | -0.007 | +0.043 | 0.891 | +0.023 |

## Promote decisions vs baseline

- **candidate_0_93**: PASS (away_improve=+0.106, home_worsen=+0.000, mae_worsen=-0.016)
- **fit_train**: PASS (away_improve=+0.212, home_worsen=+0.000, mae_worsen=-0.026)

**Recommended (counterfactual only):** fit_train

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_away_shrink_lam_screen
```

