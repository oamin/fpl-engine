Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# `score_xp_v2` rate M0 vs raw rates

Pre-registered: `protocol.score_xp_v2_rate_m0`.

Treatment applies early-season minutes pseudo-count and zero-event SE floor on `rG`/`rA` before weights/shares. Share-SE shrink stays on for both arms.

Formula: `r=G/(M+M0); Δr=max(√G,c)/(M+M0); M0=270; c=0.25; same for A; same-season GW<t only`

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed** (rates/priors GW `< t`; minutes oracle).

## Result (treatment − baseline)

- Mean Δ log-score / GW: **0.0037** [0.0006, 0.0071]
- Covers 0: **False**
- Verdict: **promote**
- Pooled player-fixture rows: 27752

| Season | n | mean Δ log | MAE raw | MAE M0 |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | 6937 | +0.0066 | 2.303 | 2.297 |
| 2023-24 | 6992 | +0.0026 | 2.341 | 2.355 |
| 2024-25 | 6931 | -0.0007 | 2.268 | 2.297 |
| 2025-26 | 6892 | +0.0063 | 2.376 | 2.371 |

## Notes

- Pool matches `score_xp_v2_screen` (minutes>60, FD AH+OU, GW5–38, σ=3).
- Positive Δ = M0 rates beat current `score_xp_v2` on Gaussian log score.
- Promoted 2026-10-16: default `RATE_M0=270`, `RATE_SE_FLOOR=0.25` in `player_rates_from_history`.
