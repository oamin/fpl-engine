# Mean-board xi_ft hist path — oracle ranks + LOSO GF/GA

Protocol: `cascade_mean_xi_ft_hist` · season **2025-26** · decision GW8 → horizon 8–19 · N=500 · seed=7

**Diagnostic.** Mean of N generative strands → one joint `xi_ft` plan (no captain in V; bank pot separate; `ft_pe_nu=0`).

## Baselines

- Club ranks: **oracle** post-GW38 finishing places
- Place→GF/GA: **rankmean±SD LOSO** (exclude scored-season E0)
- League ḡ: LOSO closed prior (`league_avg_gf`)
- Player rG/rA: **historical rate baseline** (prior seasons + same-season GW `< 8`)
- Short priced head: FD AH+OU GWs [8, 9]
- Long: oracle-rank generative (`sources=['cascade_long_oracle_rank', 'cascade_short_priced']`)

## Path summary

- Objective (Σ XI − hits): **1018.27**
- Total transfers / hits: **12** / **0**
- Terminal FT: **1** (bank_pot.in_objective=False)
- Opening rebuild free: week-0 from owned 15 with start_ft=1
- Board roll wall: 845.0s · total wall: 855.5s

## FT ledger

| GW | FT before→after | transfers | hits | XI |
| --- | ---: | ---: | ---: | ---: |
| 8 | 1→2 | 0 | 0 | 76.96 |
| 9 | 2→3 | 0 | 0 | 78.51 |
| 10 | 3→4 | 0 | 0 | 89.65 |
| 11 | 4→5 | 0 | 0 | 79.29 |
| 12 | 5→5 | 1 | 0 | 85.16 |
| 13 | 5→3 | 3 | 0 | 86.77 |
| 14 | 3→4 | 0 | 0 | 83.24 |
| 15 | 4→5 | 0 | 0 | 84.52 |
| 16 | 5→5 | 1 | 0 | 92.02 |
| 17 | 5→3 | 3 | 0 | 90.95 |
| 18 | 3→2 | 2 | 0 | 86.90 |
| 19 | 2→1 | 2 | 0 | 84.31 |

## Transfers

- GW12: buy ['Antoine Semenyo'] / sell ['Bruno Borges Fernandes'] (ft 5→5, hits=0)
- GW13: buy ['Matthijs de Ligt', 'Bruno Borges Fernandes', 'Ollie Watkins'] / sell ['João Pedro Junqueira de Jesus', 'Gabriel dos Santos Magalhães', 'Antoine Semenyo'] (ft 5→3, hits=0)
- GW16: buy ['Declan Rice'] / sell ['Bryan Mbeumo'] (ft 5→5, hits=0)
- GW17: buy ['Kevin Schade', 'Igor Thiago Nascimento Rodrigues', 'Antoine Semenyo'] / sell ['Declan Rice', 'Bruno Borges Fernandes', 'Ollie Watkins'] (ft 5→3, hits=0)
- GW18: buy ['Ryan Sessegnon', 'Mohamed Salah'] / sell ['Kevin Schade', 'Antoine Semenyo'] (ft 3→2, hits=0)
- GW19: buy ['Bryan Mbeumo', 'Bruno Borges Fernandes'] / sell ['Mohamed Salah', 'Granit Xhaka'] (ft 2→1, hits=0)

## Top mean xP (horizon average)

| player | team | pos | mean xP |
| --- | --- | --- | ---: |
| Erling Haaland | Man City | FWD | 12.12 |
| Mohamed Salah | Liverpool | MID | 8.31 |
| Tijjani Reijnders | Man City | MID | 8.18 |
| Bruno Borges Fernandes | Man Utd | MID | 7.91 |
| Bryan Mbeumo | Man Utd | MID | 7.79 |
| Yankuba Minteh | Brighton | MID | 7.63 |
| Antoine Semenyo | Bournemouth | MID | 7.39 |
| Declan Rice | Arsenal | MID | 7.02 |
| Omar Alderete | Sunderland | DEF | 7.00 |
| Granit Xhaka | Sunderland | MID | 6.92 |
| Jurriën Timber | Arsenal | DEF | 6.77 |
| Gabriel dos Santos Magalhães | Arsenal | DEF | 6.72 |
| João Pedro Junqueira de Jesus | Chelsea | FWD | 6.63 |
| Riccardo Calafiori | Arsenal | DEF | 6.52 |
| Nick Woltemade | Newcastle | FWD | 6.48 |

## Artefacts

- `data/processed/cascade_mean_xi_ft_hist/state.json`
- `data/predictions/2025-26/cascade_mean_xi_ft_hist`
- `reports/cascade_mean_xi_ft_hist.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_hist --season 2025-26 --n 500 --decision-gw 8 --horizon-end 19
```

