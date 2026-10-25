# Generative − centreline G gap dig

Season **2025-26** · OL XI GW8–19 · full-club shares · N=40 · wall 294.5s · lineup_cov=False

## Arms (Σ E[G] on XI)

| arm | Σ E[G] | resid vs G |
| --- | ---: | ---: |
| cl_fd (shrink · FD xG) | 30.07 | +2.07 |
| cl_path (shrink · path xG) | 33.11 | +5.11 |
| cl_raw (raw · path xG) | 39.11 | +11.11 |
| mean_k (raw · α · k) | 38.83 | +10.83 |
| mean_πk (dice π · α · k) | 38.83 | +10.83 |
| gen (dice) | 39.77 | +11.77 |
| actual G | 28.00 | 0 |

## Gap pieces (gen − cl_fd)

Total **+9.71** · pieces sum +9.71 (err +0.000)

| piece | Σ |
| --- | ---: |
| path λ − FD | +3.04 |
| raw sG − shrink | +6.00 |
| E[k] − xG (linear) | -0.29 |
| props π − sG | -0.00 |
| dice − πk | +0.95 |

**Dominant piece: raw_sG_vs_shrink**

## Scoreline on path (strand-mean)

- All sides: mean k 1.486 vs xG 1.493 (bias -0.007)
- XI clubs: mean k 1.947 vs xG 1.960 (bias -0.014)

## Top XI gap contributors (gen − cl_fd)

| name | pos | n | gap | path−FD | raw−sh | k−xG | props | dice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Antoine Semenyo | MID | 4 | +1.83 | +0.36 | +0.98 | +0.18 | +0.00 | +0.30 |
| Erling Haaland | FWD | 12 | +1.71 | -0.12 | +2.13 | -0.35 | +0.00 | +0.03 |
| Yankuba Minteh | MID | 9 | +1.44 | +0.37 | +0.86 | -0.01 | +0.00 | +0.22 |
| Bryan Mbeumo | MID | 7 | +1.29 | +0.38 | +1.07 | -0.08 | +0.00 | -0.08 |
| Kevin Schade | MID | 2 | +0.79 | +0.09 | +0.72 | +0.08 | +0.00 | -0.10 |
| Bruno Borges Fernandes | MID | 12 | +0.57 | +0.59 | +0.09 | -0.22 | +0.00 | +0.11 |
| Ollie Watkins | FWD | 3 | +0.57 | +0.60 | +0.09 | +0.03 | -0.00 | -0.15 |
| João Pedro Junqueira de Jesus | FWD | 4 | +0.49 | +0.09 | +0.13 | -0.22 | -0.00 | +0.49 |
| Omar Alderete | DEF | 7 | +0.48 | +0.15 | +0.33 | -0.01 | +0.00 | +0.00 |
| Riccardo Calafiori | DEF | 7 | +0.41 | +0.11 | +0.41 | +0.03 | +0.00 | -0.14 |
| Nick Woltemade | FWD | 5 | +0.36 | -0.50 | +0.63 | +0.29 | +0.00 | -0.06 |
| Gabriel dos Santos Magalhães | DEF | 11 | +0.28 | +0.15 | +0.08 | +0.03 | +0.00 | +0.01 |

## Verdict

gen−cl_fd Σ = **+9.71** on XI (gen resid +11.77, cl_fd resid +2.07). Pieces: pathλ−FD +3.04; raw−shrink +6.00; E[k]−xG -0.29; props π−sG -0.00; dice−πk +0.95. **Dominant: raw_sG_vs_shrink** (+6.00). XI-club scoreline bias k−xG -0.014.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_g_gen_centreline_gap --n 40
```

