# cascade_long place×venue club-λ screen

Pre-registered **2026-10-27** · protocol `cascade_long_place_venue_lam_screen` · wall 326.9s

## Formula

- **blind** (default): season-total place GF/GA.
- **venue_away**: `λ_H` blind; `λ_A = gf_a_pm(ra)·((GA_H_blind/38)/ḡ)`.
- **venue**: EV-matched full place×venue.
- **naive**: isolate diagnostic (blended ḡ; overshoots home).
- **blend**: `λ_A ← (1−w)·blind + w·venue_away`.

## Locked before holdout

- Baseline: `venue_mode=blind`.
- Fit blend w on 2022-23, 2023-24, 2024-25 → **`w=0.90`**.
- Holdout: **2025-26** GW8–38.
- OOS: 2022-23, 2023-24, 2024-25.
- Promote if vs baseline: away |bias| ↓ ≥0.1, home |bias| ↑ ≤0.05, MAE ↑ ≤0.02.
- Default `DEFAULT_VENUE_MODE` **not** flipped here (needs OOS + PI re-approval).

## Holdout 2025-26 (Opta λ − actual)

| arm | mode | w | bias | bias H | bias A | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline_blind | blind | 0.00 | +0.125 | -0.007 | +0.256 | 0.917 |
| venue_away | venue_away | 1.00 | +0.055 | -0.007 | +0.117 | 0.903 |
| venue_ev | venue | 1.00 | +0.125 | +0.131 | +0.119 | 0.923 |
| naive | naive | 1.00 | +0.137 | +0.282 | -0.007 | 0.930 |
| blend_fit | blend | 0.90 | +0.062 | -0.007 | +0.131 | 0.904 |

## Promote decisions vs baseline

- **venue_away**: PASS (away_improve=+0.139, home_worsen=+0.000, mae_worsen=-0.015)
- **venue_ev**: FAIL (away_improve=+0.137, home_worsen=+0.124, mae_worsen=+0.006)
- **naive**: FAIL (away_improve=+0.248, home_worsen=+0.275, mae_worsen=+0.013)
- **blend_fit**: PASS (away_improve=+0.125, home_worsen=+0.000, mae_worsen=-0.014)

## OOS prior seasons

- **2022-23**: winners=['venue_away', 'venue_ev', 'blend_fit']
- **2023-24**: winners=['none']
- **2024-25**: winners=['none']

### OOS pass counts (of 3)

- venue_away: 1/3
- venue_ev: 1/3
- naive: 0/3
- blend_fit: 1/3

**Recommended (counterfactual only):** venue_away

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_place_venue_lam_screen
```

