# cascade_long as-of expanding club H/A λ screen

Pre-registered **2026-10-29** · protocol `cascade_long_expanding_ha_lam_screen` · wall 14.9s

## Formula

- **blind**: Opta place season-total GF/GA (default).
- **expanding_ha**: as-of club home/away GF/GA (prior seasons + current before GW; 10 pseudo prior) × opp venue defence / league venue mean.
- **expanding_away**: `λ_H` blind; `λ_A` from expanding rates.
- **blend**: `(1−w)·blind + w·expanding_ha`.

## Holdout 2025-26

| arm | bias | bias H | bias A | MAE |
| --- | ---: | ---: | ---: | ---: |
| baseline_blind | +0.122 | -0.015 | +0.259 | 0.918 |
| expanding_ha | +0.049 | +0.067 | +0.031 | 0.916 |
| expanding_away | +0.008 | -0.015 | +0.031 | 0.905 |
| blend_fit | +0.100 | +0.010 | +0.191 | 0.906 |

## Promote vs baseline

- **expanding_ha**: FAIL (away_improve=+0.228, home_worsen=+0.052, mae_worsen=-0.002)
- **expanding_away**: PASS (away_improve=+0.228, home_worsen=+0.000, mae_worsen=-0.013)
- **blend_fit**: FAIL (away_improve=+0.068, home_worsen=-0.005, mae_worsen=-0.012)

## OOS

- **2022-23**: winners=['expanding_ha', 'expanding_away']
- **2023-24**: winners=['none']
- **2024-25**: winners=['none']
- **2025-26**: winners=['expanding_away']

### OOS pass counts

- expanding_ha: 1/4
- expanding_away: 2/4
- blend_fit: 0/4

**Recommended:** none — keep venue_mode=blind

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_expanding_ha_lam_screen
```

