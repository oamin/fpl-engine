Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Cascade_short generative goal shares π∝r vs π∝r·m (+ L3)

Pre-registered: `protocol.cascade_short_goal_weight_likelihood_screen`.

Treatment: cascade_short generative MC-mean with ``goal_weight=rate`` (π∝r). Baseline: ``goal_weight=rate_minutes``. Reference: FD centreline ``score_xp_v2`` (L3/L4 mean layer) on the same fits.

N_MC=6 · GW8–38 · wall 389.9s

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Result (treatment − baseline = π∝r − rate_minutes)

- Mean Δ log-score / GW: **+0.0080** [-0.0022, +0.0170]
- Covers 0: **True**
- Verdict: **inconclusive**
- Pooled player-GW rows: 23302
- Default `goal_weight=rate_minutes` **not** flipped here.

## vs L3 reference (diagnostic)

- rate_minutes − L3: **-0.0384** [-0.0445, -0.0321]
- π∝r − L3: **-0.0304** [-0.0368, -0.0244]

| Season | n | mean Δ log | MAE rm | MAE rate | MAE L3 | rm−L3 | rate−L3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 5787 | +0.0057 | 2.491 | 2.469 | 2.428 | -0.0340 | -0.0283 |
| 2023-24 | 5737 | -0.0022 | 2.478 | 2.478 | 2.411 | -0.0326 | -0.0348 |
| 2024-25 | 5866 | +0.0171 | 2.446 | 2.387 | 2.339 | -0.0491 | -0.0320 |
| 2025-26 | 5912 | +0.0109 | 2.610 | 2.560 | 2.522 | -0.0376 | -0.0267 |

## Notes

- Positive Δ = π∝r beats rate_minutes on Gaussian log score.
- Same weight formula as parked `score_xp_v2_goal_share_rate`, but on **cascade_short generative** dice (not centreline `score_team_side`).
- Eligible pool (not path XI): minutes>60, ≥3 prior apps, FD-priced weeks.
- Dig B XI resid (+100 vs +14) is a different metric; this is the gate.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_short_goal_weight_likelihood_screen
```
