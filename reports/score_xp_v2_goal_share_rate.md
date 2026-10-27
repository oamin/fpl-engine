Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# `score_xp_v2` goal shares π∝r vs π∝r·m

Pre-registered: `protocol.score_xp_v2_goal_share_rate`.

Treatment drops minutes from **goal** share weights (``π ∝ rG`` among m>0). Baseline keeps retired ``w = rG·m/90``. Assists stay ``∝ rA·m/90`` on both arms. Share-SE position×minutes prior on for both.

Formula: `treatment wG=rG (m gates club only); baseline wG=rG·m/90; both π=w/Σw then share-SE position×minutes; assists ∝rA·m/90 both arms`

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed** (rates/priors GW `< t`; minutes oracle).

## Result (treatment − baseline)

- Mean Δ log-score / GW: **-0.0005** [-0.0010, -0.0001]
- Covers 0: **False**
- Verdict: **park_clear_lose**
- Pooled player-fixture rows: 27752

| Season | n | mean Δ log | MAE rm | MAE rate |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | 6937 | -0.0008 | 2.279 | 2.268 |
| 2023-24 | 6992 | -0.0005 | 2.321 | 2.308 |
| 2024-25 | 6931 | +0.0002 | 2.264 | 2.246 |
| 2025-26 | 6892 | -0.0011 | 2.353 | 2.345 |

## Notes

- Pool matches `score_xp_v2_screen` (minutes>60, FD AH+OU, GW5–38, σ=3).
- Positive Δ = π∝r beats π∝r·m on Gaussian log score.
- Cascade OL XI resid dig (N=40): +10.2 → +0.2; this screen is the gated player-GW likelihood.
- **Action 2026-10-27:** park — default stays `goal_weight='rate_minutes'` (`w=r·m/90`).
