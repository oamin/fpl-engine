# away_shrink s=0.75 screen arm

Add-on to `cascade_long_away_shrink_lam_screen` · wall 6.5s

Motive: XI resid G/CS both ≈0 near **s=0.75** on the GW8–38 Opta π∝r trend.
Same promote rule vs baseline `s=1.0`. Default `AWAY_SHRINK` **not** flipped.

## Holdout 2025-26 club λ (Opta − actual)

| arm | s | bias | bias H | bias A | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_1_00 | 1.00 | +0.125 | -0.007 | +0.256 | 0.917 |
| candidate_0_93 | 0.93 | +0.072 | -0.007 | +0.150 | 0.902 |
| fit_train_0_86 | 0.86 | +0.018 | -0.007 | +0.043 | 0.891 |
| candidate_0_75 | 0.75 | -0.065 | -0.007 | -0.123 | 0.886 |

**Promote s=0.75 vs baseline:** PASS (away_improve=+0.132, home_worsen=+0.000, mae_worsen=-0.031; thresholds away≥0.1, home≤0.05, mae≤0.02)

## XI counterfactual (already run)

| s | resid G | resid CS |
| ---: | ---: | ---: |
| 1.00 | +32.5 | -29.2 |
| 0.86 | +12.1 | -11.1 |
| **0.75** | **-2.4** | **+2.4** |

## Reproduce

```bash
# λ arm (this report)
PYTHONPATH=. python3 -c "from src.eval.cascade_long_away_shrink_lam_screen import *"
# XI map
PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp --n 40 --goal-weight rate --match-source opta --away-shrink 0.75 --steps 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38 --decision-gw-only
```

