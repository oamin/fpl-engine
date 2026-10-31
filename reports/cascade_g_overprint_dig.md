# G overprint dig — pot vs share vs selection

Season **2025-26** · OL XI GW8–19 · full-club shares · centreline decomp · gen N=40 · wall 114.6s

## Identity

On each XI player-GW (centreline, full-club `sG`):

`resid = E[G] − G = sG·α_G·(xG − GF) + (sG·α_G·GF − G)`

- **pot piece** Σ = **-2.89**
- **share piece** Σ = **+4.96**
- **total** Σ = **+2.07** (identity err +0.000)
- generative Σ resid (side check) = **+7.33**

|Pot| vs |share| mass: pot 37% of |pot|+|share|.

## Team-GW pot (xG − GF)

| panel | n | mean xG | mean GF | bias |
| --- | ---: | ---: | ---: | ---: |
| League GW8–19 | 240 | 1.415 | 1.433 | -0.018 |
| XI clubs | 79 | 1.712 | 1.848 | -0.136 |

## Selection bootstrap (MID/FWD/DEF resid)

- XI mean resid **+0.017** · pool n=1968 · boot p05/p95 -0.028/+0.073 · p_right=0.616

## Top XI resid contributors

| name | pos | n | ΣE[G] | ΣG | resid | pot | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| João Pedro Junqueira de Jesus | FWD | 4 | 2.86 | 1 | +1.86 | -0.27 | +2.13 |
| Yankuba Minteh | MID | 9 | 1.83 | 0 | +1.83 | -0.09 | +1.92 |
| Nick Woltemade | FWD | 5 | 2.74 | 1 | +1.74 | -0.21 | +1.95 |
| Kevin Schade | MID | 2 | 0.76 | 0 | +0.76 | -0.20 | +0.95 |
| Riccardo Calafiori | DEF | 7 | 0.49 | 0 | +0.49 | +0.04 | +0.46 |
| Omar Alderete | DEF | 7 | 0.42 | 0 | +0.42 | -0.05 | +0.47 |
| Granit Xhaka | MID | 10 | 1.36 | 1 | +0.36 | -0.09 | +0.45 |
| Jurriën Timber | DEF | 5 | 0.28 | 0 | +0.28 | -0.02 | +0.30 |
| Marcos Senesi Barón | DEF | 7 | 0.22 | 0 | +0.22 | +0.06 | +0.16 |
| Declan Rice | MID | 1 | 0.22 | 0 | +0.22 | +0.06 | +0.16 |
| Matthijs de Ligt | DEF | 1 | 0.08 | 0 | +0.08 | +0.05 | +0.04 |
| Joachim Andersen | DEF | 1 | 0.04 | 0 | +0.04 | +0.01 | +0.03 |

## Verdict

**Not pot.** XI clubs' FD xG ≤ GF (team-GW bias -0.136; pot piece -2.9). Centreline resid is small (+2.1) and **share-driven** (share piece +5.0): sG mass on XI players who blanked. Generative resid stays higher (+7.3) — dice/nonlinear gap vs centreline, not FD λ mis-fit. XI selection resid is not in the bootstrap tail.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_g_overprint_dig --n 40
```

