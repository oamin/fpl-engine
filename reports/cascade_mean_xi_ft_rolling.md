# Rolling mean xi_ft — stick to the open-loop plan?

Protocol: `cascade_mean_xi_ft_rolling` · season **2025-26** · horizon 8–19 · steps [8, 9, 10] · N=100 · seed=7

**Diagnostic.** Open-loop mean-board `xi_ft` at GW8; commit that week-0, then at GW9/GW10 refresh as-of rates + mean boards and replan from the committed 15 (receding stick test).

## Baselines (each step)

- Oracle post-GW38 ranks; LOSO rankmean GF/GA; prior rate baseline + same-season GW `< t`
- Mean of N generative strands → one `xi_ft` plan (value_captain=false, ft_pe_nu=0)
- Wall: 501.1s

## Open-loop (GW8 → 19)

- Objective: **1018.95** · transfers/hits **12/0** · terminal FT 1
- Board roll: 168.5s · sources ['cascade_long_oracle_rank', 'cascade_short_priced']

| GW | FT before→after | transfers | hits | XI |
| --- | ---: | ---: | ---: | ---: |
| 8 | 1→2 | 0 | 0 | 75.93 |
| 9 | 2→3 | 0 | 0 | 77.64 |
| 10 | 3→4 | 0 | 0 | 91.49 |
| 11 | 4→5 | 0 | 0 | 79.48 |
| 12 | 5→5 | 1 | 0 | 83.43 |
| 13 | 5→4 | 2 | 0 | 88.11 |
| 14 | 4→3 | 2 | 0 | 80.36 |
| 15 | 3→4 | 0 | 0 | 83.30 |
| 16 | 4→4 | 1 | 0 | 92.95 |
| 17 | 4→2 | 3 | 0 | 94.60 |
| 18 | 2→2 | 1 | 0 | 83.48 |
| 19 | 2→1 | 2 | 0 | 88.18 |

### Open-loop transfers (full)

- GW12: buy ['Antoine Semenyo'] / sell ['Bryan Mbeumo'] (ft 5→5)
- GW13: buy ['Kevin Schade', 'Ollie Watkins'] / sell ['João Pedro Junqueira de Jesus', 'Antoine Semenyo'] (ft 5→4)
- GW14: buy ['Bryan Mbeumo', 'Joachim Andersen'] / sell ['Kevin Schade', 'Jurriën Timber'] (ft 4→3)
- GW16: buy ['Declan Rice'] / sell ['Bryan Mbeumo'] (ft 4→4)
- GW17: buy ['Kevin Schade', 'Richarlison de Andrade', 'Antoine Semenyo'] / sell ['Yankuba Minteh', 'Declan Rice', 'Ollie Watkins'] (ft 4→2)
- GW18: buy ['Bryan Mbeumo'] / sell ['Kevin Schade'] (ft 2→2)
- GW19: buy ['Yankuba Minteh', 'Matthijs de Ligt'] / sell ['Granit Xhaka', 'Riccardo Calafiori'] (ft 2→1)

## Rolling commits (GW8 → GW10)

| step | commit transfers | hits | FT before→after | owned Jaccard vs OL | sells match | buys match | replan future transfers |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| GW8 | 0 | 0 | 1→2 | 1.00 | True | True | 12 |
| GW9 | 1 | 0 | 2→2 | 0.88 | False | False | 11 |
| GW10 | 1 | 0 | 2→2 | 0.76 | False | False | 10 |

- Mean owned Jaccard (steps): **0.880**
- Transfer-set exact match rate: **0.33** (1/3)
- Rolling committed transfers / hits over steps: **2/0**
- Open-loop transfers / hits on the same steps: **0/0**

## Commit narrative

- **GW8:** buy [] / sell [] · ft 1→2 · hits=0 · board 0.0s · pool=157
  - replan still wants later: GW12 buy ['Antoine Semenyo']/sell ['Bryan Mbeumo'], GW13 buy ['Kevin Schade', 'Ollie Watkins']/sell ['João Pedro Junqueira de Jesus', 'Antoine Semenyo'], GW14 buy ['Bryan Mbeumo', 'Joachim Andersen']/sell ['Kevin Schade', 'Jurriën Timber'], GW16 buy ['Declan Rice']/sell ['Bryan Mbeumo'] …
- **GW9:** buy ['Michael Keane'] / sell ['Marcos Senesi Barón'] · ft 2→2 · hits=0 · board 153.3s · pool=158
  - replan still wants later: GW9 buy ['Michael Keane']/sell ['Marcos Senesi Barón'], GW11 buy ['Igor Thiago Nascimento Rodrigues', 'Nordi Mukiele', 'Antoine Semenyo']/sell ['Granit Xhaka', 'Riccardo Calafiori', 'Nick Woltemade'], GW13 buy ['Morgan Rogers']/sell ['Antoine Semenyo'], GW14 buy ['Antoine Semenyo']/sell ['Morgan Rogers'] …
- **GW10:** buy ['Ryan Sessegnon'] / sell ['Yankuba Minteh'] · ft 2→2 · hits=0 · board 137.3s · pool=152
  - replan still wants later: GW10 buy ['Ryan Sessegnon']/sell ['Yankuba Minteh'], GW12 buy ['Danny Welbeck', 'Nordi Mukiele', 'Antoine Semenyo']/sell ['Granit Xhaka', 'Nick Woltemade', 'Jurriën Timber'], GW14 buy ['Amad Diallo', 'Marcos Senesi Barón']/sell ['Ryan Sessegnon', 'Riccardo Calafiori'], GW16 buy ['Ryan Sessegnon', 'Morgan Rogers']/sell ['Bruno Borges Fernandes', 'Amad Diallo'] …

## Verdict

**Soft drift, same regime.** GW8 matches open-loop (hold). After forecast refresh, GW9 sells Senesi→Keane and GW10 sells Minteh→Sessegnon — one free transfer each — while open-loop still banks (0 transfers through GW11). Ownership Jaccard 1.00→0.88→0.76 (mean 0.88). Still **0 hits**; FT bank stays healthy (ends GW10 at 2). The long-horizon transfer calendar is rewritten at each replan, not merely delayed.

## Artefacts

- `data/processed/cascade_mean_xi_ft_rolling/state.json`
- `data/predictions/2025-26/cascade_mean_xi_ft_rolling`
- `reports/cascade_mean_xi_ft_rolling.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_rolling --season 2025-26 --n 100 --steps 8,9,10
```

