# Stepped ≥60′ selection pool — xP(G, CS, GA) vs actual

Season **2025-26** · horizon 8–19 · commit steps [8, 9, 10, 11, 12] · N=40 · seed=7 · wall 560.5s

## Protocol

- At each decision GW t: rebuild **selection** pool = players with as-of mean minutes > 60 (and ≥2 apps) before t.
- **Share** pool = full club with m>0 (impact subs on generative dice).
- That as-of state (rates, minutes, both pools) is **projected forward** on strands for t…horizon end until the next step refreshes it.
- Mean of N generative boards → `xi_ft` plan; commit week-t XI.
- Residual = xP − actual on that XI. **GA** = goals-against points (`xp_gc` / FPL GC).

![per-match resid](/opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid.png)

## Overall (committed GW8–12 XI)

- Player-GWs: **55**

| component | Σ xP | Σ actual | Σ resid | mean / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 65.8 | 42.0 | +23.8 | +0.434 | 1.220 |
| CS | 43.1 | 45.0 | -1.9 | -0.035 | 0.835 |
| GA | -6.3 | -7.0 | +0.7 | +0.013 | 0.183 |

## Per committed GW

| GW | select n | share n | xfers | hits | r_G | r_CS | r_GA | ΣxP_G | ΣP_G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 157 | 379 | 1 | 0 | -0.18 | -0.84 | -0.03 | 10.1 | 12.0 |
| 9 | 158 | 377 | 1 | 0 | +1.45 | -0.41 | -0.15 | 16.0 | 0.0 |
| 10 | 152 | 381 | 1 | 0 | +0.05 | -0.07 | -0.10 | 13.6 | 13.0 |
| 11 | 158 | 379 | 1 | 0 | -0.10 | +0.67 | +0.37 | 11.9 | 13.0 |
| 12 | 157 | 370 | 0 | 0 | +0.94 | +0.47 | -0.03 | 14.4 | 4.0 |

## Commits

- **GW8:** buy ['Nick Pope'] / sell ['David Raya Martín'] · ft 1→1 · hits=0
- **GW9:** buy ['Nordi Mukiele'] / sell ['Omar Alderete'] · ft 1→1 · hits=0
- **GW10:** buy ['Dominik Szoboszlai'] / sell ['Kevin Schade'] · ft 1→1 · hits=0
- **GW11:** buy ['Igor Thiago Nascimento Rodrigues'] / sell ['Nick Woltemade'] · ft 1→1 · hits=0
- **GW12:** buy [] / sell [] · ft 1→2 · hits=0

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_step_pool_resid --n 40 --steps 8,9,10,11,12
```

