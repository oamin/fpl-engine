Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# `score_xp_v2` minutes decay vs flat last-3

Pre-registered: `protocol.score_xp_v2_minutes_decay`.

Both arms use **predicted** minutes from GW `< t` (not oracle). Treatment: geometric decay ρ=0.7 on last 3 GW rows (zeros included). Baseline: flat mean of the same window.

Formula: `m̂=Σ ρ^{n-1-i} m_{t-n+i} / Σ ρ^{n-1-i}; n≤3; ρ=0.7; rates/priors/share-SE/M0 unchanged`

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Result (treatment − baseline)

- Mean Δ log-score / GW: **-0.0009** [-0.0026, 0.0007]
- Covers 0: **True**
- Verdict: **inconclusive**
- Pooled player-fixture rows: 27752

| Season | n | mean Δ log | MAE flat | MAE decay |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | 6937 | -0.0009 | 2.395 | 2.397 |
| 2023-24 | 6992 | +0.0007 | 2.458 | 2.455 |
| 2024-25 | 6931 | -0.0005 | 2.400 | 2.402 |
| 2025-26 | 6892 | -0.0027 | 2.539 | 2.540 |

## Notes

- Pool: actual minutes>60, FD AH+OU, GW5–38, σ=3 (same as score_xp_v2_screen).
- Positive Δ = decay beats flat predicted minutes on Gaussian log score.
- Parked 2026-10-16: keep flat last-3 (ρ=1) for live / L4 projections.
