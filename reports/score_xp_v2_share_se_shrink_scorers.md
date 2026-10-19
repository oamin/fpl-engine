Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# `score_xp_v2` share-SE shrink (scorers-only prior) vs raw

Pre-registered: `protocol.score_xp_v2_share_se_shrink_scorers`.

Treatment shrinks **sG/sA** toward a **minutes prior on raw `s>0` only** (non-scorers stay at share 0). Fixes DEF/GK goal-share leak from the 2026-10-16 all-minutes prior.

Formula: `s0=m/Σm on s>0 only; zeros stay 0; τ²=median(Δs²); K=τ²/(τ²+Δs²); s̃=Ks+(1−K)s0; renormalise`

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed** (rates/priors GW `< t`; minutes oracle).

## Result (scorers prior − raw)

- Mean Δ log-score / GW: **-0.0036** [-0.0066, -0.0004]
- Covers 0: **False**
- Verdict: **park_clear_lose**
- Pooled player-fixture rows: 27760

## Diagnostic (scorers prior − legacy all-minutes)

- Mean Δ log-score / GW: **-0.0103** [-0.0134, -0.0072]

| Season | n | Δ vs raw | Δ vs legacy | MAE raw | MAE scorers | MAE legacy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 6937 | -0.0055 | -0.0127 | 2.295 | 2.324 | 2.297 |
| 2023-24 | 6992 | -0.0072 | -0.0099 | 2.330 | 2.364 | 2.355 |
| 2024-25 | 6931 | +0.0009 | -0.0069 | 2.271 | 2.296 | 2.297 |
| 2025-26 | 6900 | -0.0026 | -0.0118 | 2.384 | 2.397 | 2.375 |

## Notes

- Pool matches `score_xp_v2_screen` (minutes>60, FD AH+OU, GW5–38, σ=3).
- Promote gate is vs raw only; legacy column is diagnostic.
- Positive Δ = scorers-prior shrink beats the named baseline on Gaussian log score.
- **Action 2026-10-17:** park_clear_lose → default `share_se_shrink=False` (raw shares). Legacy all-minutes prior also parked (DEF/GK leak). Code keeps `prior_support='positive_share'|'all_minutes'` for future screens.
