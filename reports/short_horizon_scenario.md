# Short-horizon scenario layer (priced t : t+1)

Advisor layer on market-priced weeks. Does **not** replace published `score_xp`. Shares frozen at the deadline; λ sampled inside ±15% of opening Shin pots; path FT between t and t+1.

## Decision

**keep_scenario_advisor**

- Keep advisor if path stick ≥ independent t∩t+1 overlap + 0.05. Published `score_xp` stays the centreline mean layer either way.
- path_signal=True · mean_layer_quiet=True (|ΔxP|<0.25)

## Summary

| season | 2024-25 |
| deadlines ok | 3 / 3 |
| mean ΔV (path − indep) | -6.627 |
| mean stick (path) | 0.897 |
| mean indep overlap t∩t+1 | 0.244 |
| mean hits t+1 | 2.17 |
| mean |ΔxP| vs centreline (t) | 0.2062 |

## Per deadline

| GW | V_indep | mean V_path | ΔV | stick | hits | |ΔxP| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 112.42 | 109.92 | -2.50 | 0.93 | 0.00 | 0.1877 |
| 20 | 121.37 | 117.13 | -4.25 | 0.93 | 0.00 | 0.2100 |
| 28 | 114.92 | 101.79 | -13.13 | 0.83 | 6.50 | 0.2209 |

## Read

1. Small mean |ΔxP| under sampling → centreline mean layer stays the published score.
2. Higher path stick vs independent week rebuilds → scenario+FT layer is the short-horizon analogue of multistage item 1b.
3. Negative ΔV is expected (hit cost); not a kill by itself.
4. Hand-off of strand rates into t+2 Opta cascade is deferred.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.models.short_horizon_cascade
```
