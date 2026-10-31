# cascade_long FD venue mean-ratio λ screen

Pre-registered **2026-10-29** · protocol `cascade_long_fd_venue_scale_lam_screen` · wall 16.0s

## Lever

Venue dominates the Opta−FD residual. Scale blind Opta by venue:

```
λ'_H = r_H · λ_H
λ'_A = r_A · λ_A
r_ha = mean(FD_ha) / mean(Opta_ha)   # train fit
```

Train-fit: **r_H=1.1440**, **r_A=0.9288**.

Prior: margin-shrink FAIL (does not move venue means when E[λ_H]≈E[λ_A] under blind).

## Holdout 2025-26

| arm | r_H | r_A | bias | bias H | bias A | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_blind | 1.000 | 1.000 | +0.121 | -0.013 | +0.255 | 0.918 |
| fd_venue_fit | 1.144 | 0.929 | +0.175 | +0.204 | +0.146 | 0.917 |
| fd_venue_away_only | 1.000 | 0.929 | +0.066 | -0.013 | +0.146 | 0.902 |
| pooled_const | 1.115 | 0.893 | +0.126 | +0.160 | +0.092 | 0.908 |

## Promote vs baseline

- **fd_venue_fit**: FAIL (away_improve=+0.108, home_worsen=+0.191, mae_worsen=-0.001)
- **fd_venue_away_only**: PASS (away_improve=+0.108, home_worsen=+0.000, mae_worsen=-0.016)
- **pooled_const**: FAIL (away_improve=+0.163, home_worsen=+0.146, mae_worsen=-0.010)

## OOS

- **2022-23**: winners=['fd_venue_away_only', 'pooled_const']
- **2023-24**: winners=['none']
- **2024-25**: winners=['none']
- **2025-26**: winners=['fd_venue_away_only']

### OOS pass counts

- fd_venue_fit: 0/4
- fd_venue_away_only: 2/4
- pooled_const: 1/4

**Recommended:** none — keep r_H=r_A=1 (blind)

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_fd_venue_scale_lam_screen
```

