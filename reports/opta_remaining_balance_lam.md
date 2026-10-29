# Remaining-season balance λ vs blind Opta

Hypothesis: place-rank implies season GF/GA. Blind uses `place/38` every week. If a club is behind on G / soft on GC so far, remaining λ should run hotter.

```
gf_rem = (place_GF − ΣG_so_far) / games_left
λ_H = gf_rem_H · (ga_rem_A / league_mean_gf_rem)
```

As-of before GW; LOSO place targets; rates clipped [0.05, 5].

## Pooled vs actual

| arm | bias | MAE | RMSE | H | A | corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blind Opta | -0.010 | 0.949 | 1.201 | -0.140 | +0.120 | 0.314 |
| remaining-balance | -0.009 | 1.047 | 1.351 | -0.133 | +0.115 | 0.194 |
| FD | +0.013 | 0.942 | 1.183 | +0.043 | -0.017 | 0.351 |

corr(GF behind pace, act−blind) = **-0.041**

## By season MAE

| season | blind | balance | Δ |
| --- | ---: | ---: | ---: |
| 2022-23 | 0.976 | 1.062 | +0.086 |
| 2023-24 | 0.964 | 1.094 | +0.130 |
| 2024-25 | 0.936 | 1.015 | +0.079 |
| 2025-26 | 0.918 | 1.017 | +0.099 |

## Verdict

- Remaining-balance MAE **1.047** vs blind **0.949** (Δ **+0.098**).
- Pace→underprint signal corr = **-0.041**.

