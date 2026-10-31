# Why cascade_short generative loses to L3

Pre-registered **2026-10-29** · protocol `cascade_gen_vs_l3_loss_anatomy` · wall 0.1s · n=23302

## Verdict

- **Not a mean-bias gap.** rate_minutes bias **-0.963** vs L3 **-0.960** (both cold vs total points).
- Gen has **higher pred SD** (1.81 vs 1.57) and **lower corr** (0.217 vs 0.237).
- Loss concentrates on **high-xp** players (top L3 quintile).
- Aligns with viability B1 (**gen clearly_loses** vs centreline). Gen-as-mean stays parked; do not chase XI resid G via more dice noise.

## Arm summary

| arm | bias | MAE | corr | std(xp) | mean log |
| --- | ---: | ---: | ---: | ---: | ---: |
| rate_minutes | -0.963 | 2.507 | 0.217 | 1.814 | -2.7210 |
| rate | -1.014 | 2.474 | 0.223 | 1.728 | -2.7124 |
| l3 | -0.960 | 2.425 | 0.237 | 1.569 | -2.6820 |

## Δ log (rate_minutes − L3) by L3-xp quintile

| quintile | n | Δ log | bias rm | bias L3 |
| --- | ---: | ---: | ---: | ---: |
| (-1.4069999999999998, 1.26] | 4661 | -0.0076 | -2.265 | -2.278 |
| (1.26, 2.278] | 4660 | -0.0239 | -1.617 | -1.622 |
| (2.278, 3.152] | 4660 | -0.0311 | -0.522 | -0.519 |
| (3.152, 3.943] | 4660 | -0.0523 | -0.321 | -0.287 |
| (3.943, 15.408] | 4661 | -0.0797 | -0.088 | -0.093 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_gen_vs_l3_loss_anatomy
```

