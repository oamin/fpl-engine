# Horizon forecast uncertainty — Stage 1 screen

## ChatGPT proposal — what to keep / defer

### Worth trialing now

- Separate match-outcome noise from forecast/strength uncertainty
- Calibrate lead-time growth from historical errors (Stage 1)
- Prefer rolling-origin / as-of rebuilds over hindsight Opta ranks
- Start with league-wide drift params; team-specific only if needed
- Do not apply an arbitrary %/GW inflation

### Defer

- Stage 2 random-walk / mean-reverting strength drift — Stage 1 slope is tiny vs Poisson
- Stage 3 nested MC (scenario λ then score) until material drift is shown
- Team-specific q_A / q_D
- Archived Opta forecast reconstruction (we lack mid-season Opta snaps)
- Arbitrary % inflation of GF/GA or λ per future gameweek

### Already in the engine (do not double-count)

- Opta rank→season GF/GA OLS ±MAE (level uncertainty, not time-scaled)
- Match pots via Dixon–Coles / Poisson in xp_on_pot (match noise)
- OUTRIGHT_SHRINK=0.5 for unpriced weeks (level shrink, not lead curve)

## Question

Does GF/GA forecast error grow with lead time after removing Poisson match noise (excess MSE = MSE − mean λ)?

No archived Opta mid-season tables. Proxies: (1) rolling expanding GF/GA aged by lead *h*; (2) rates frozen after GW6 (sticky Opta-like).
Closed seasons only: 2022/23, 2023/24, 2024/25, 2025/26. 2026/27 holdout unused.

## Rolling expanding rates

n team-matches×leads = **12756**.

| lead (GW) | n | MAE | RMSE | MSE | mean λ | excess MSE | Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2798 | 0.955 | 1.205 | 1.452 | 1.460 | -0.008 | 0.303 |
| 2 | 2718 | 0.956 | 1.203 | 1.448 | 1.460 | -0.012 | 0.296 |
| 4 | 2556 | 0.959 | 1.203 | 1.447 | 1.459 | -0.012 | 0.300 |
| 6 | 2422 | 0.962 | 1.211 | 1.467 | 1.457 | 0.010 | 0.280 |
| 8 | 2262 | 0.969 | 1.220 | 1.489 | 1.458 | 0.031 | 0.264 |

Growth fit `excess_mse ~ a + b·lead`: b=0.00582 /GW, R²=0.7995, Δexcess(H8−H1)=0.0392, ΔMAE=0.0138, stat_positive=True, material=False.

### LOSO (leave one season out)

| holdout | excess slope /GW | R² | material |
|---|---:|---:|:---:|
| 2022/23 | 0.01078 | 0.887 | False |
| 2023/24 | 0.00468 | 0.8326 | False |
| 2024/25 | 0.00311 | 0.4761 | False |
| 2025/26 | 0.00465 | 0.6523 | False |

## Freeze after GW6 (Opta-sticky proxy)

n = **2562**. Leads binned to [1, 2, 4, 6, 8]. Freeze lead bins are unbalanced (most mass in lead≥7); treat freeze slope as diagnostic only.

| lead (GW) | n | MAE | RMSE | MSE | mean λ | excess MSE | Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62 | 0.940 | 1.187 | 1.408 | 1.427 | -0.019 | 0.345 |
| 2 | 74 | 0.891 | 1.106 | 1.223 | 1.427 | -0.204 | 0.409 |
| 4 | 160 | 0.857 | 1.103 | 1.216 | 1.436 | -0.220 | 0.457 |
| 6 | 158 | 1.021 | 1.245 | 1.549 | 1.428 | 0.121 | 0.118 |
| 8 | 2108 | 0.974 | 1.239 | 1.534 | 1.434 | 0.100 | 0.243 |

Growth fit: b=0.03508 /GW, R²=0.3839, material=False.

## Verdict

- Rolling growth statistically positive: **True**
- Rolling growth material for Stage 2: **False**
- Freeze growth material: **False**

Park Stage 2–3. Rolling excess MSE rises only ~0.04 from H=1 to H=8 against MSE≈1.45 (mostly Poisson). Keep Opta ±MAE as *level* uncertainty; do not invent a %/GW lead penalty. Revisit shared-q drift only if a richer as-of strength archive (Opta snaps or priced long-horizon markets) shows material growth.

## Reproduce

```bash
PYTHONPATH=. python3 scripts/screen_horizon_forecast_uncertainty.py
```

