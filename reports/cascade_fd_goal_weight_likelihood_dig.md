# Cascade FD goal_weight likelihood dig

Pre-registered **2026-10-28** · protocol `cascade_fd_goal_weight_likelihood_dig` · wall 0.4s

## Verdict

- XI resid G: π∝r **+14.0** vs rate_minutes **+100.3** (minutes hotter).
- Mean log score on G (σ=3.0): π∝r **-2.2469** vs rate_minutes **-2.2622**.
- Paired GW Δ (rate − rate_minutes) on G: **+0.0153** [-0.0016, +0.0322] · covers 0: **True** · prefer **rate**.
- Gated centreline reference (`score_xp_v2_goal_share_rate`): **park_clear_lose** for π∝r (mean Δ -0.0005).
- Default `goal_weight=rate_minutes` **unchanged**.

## Position resid G (diagnostic)

| pos | π∝r resid G | rate_minutes resid G | π∝r mean log G | rm mean log G |
| --- | ---: | ---: | ---: | ---: |
| GKP | +0.0 | +0.0 | -2.0176 | -2.0176 |
| DEF | +4.2 | +28.7 | -2.1337 | -2.1497 |
| MID | -4.9 | +24.8 | -2.2134 | -2.2140 |
| FWD | +14.7 | +46.9 | -2.5199 | -2.5580 |

## Notes

- Log scores here are on **component** G (and G+CS+GA vs P_sum), not full `score_xp_v2` totals — diagnostic for the cascade generative map.
- Hotter mean resid can still win likelihood if variance/calibration improves; compare Δ log, not Σ resid alone.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_fd_goal_weight_likelihood_dig
```

