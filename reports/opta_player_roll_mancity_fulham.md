# Opta place → player rolls: Man City v Fulham

Trial: `python3 -m src.markets.opta_player_roll_trial --n 2000`

## Match baseline (cascade_long)

| | Man City | Fulham |
|---|---:|---:|
| Opta place | 1 | 20 |
| Season GF/GA centre | 77.6 / 38.2 | 34.6 / 74.1 |
| Centre λ | 2.911 | 0.574 |
| Sampled mean λ | 2.913 | 0.574 |

Each path: sample season GF/GA inside rank ±MAE → λ → `M` → `k ∼ M` → lineup cov → attack dice → CS/GC from rolled GA. **Props off.**

## Aggregate (N=2000)

- Mean scoreline **2.93–0.58**
- P(CS City) **0.571**, P(CS Fulham) **0.056**
- Top scorelines: 2-0, 3-0, 4-0, 1-0, 2-1, 3-1

### Top scorers (goal events)

Haaland 2228 · Cherki 1225 · Semenyo 940 · King (FUL) 572 · Guéhi 469

### Top mean xP (path average)

Haaland 6.86 · Cherki 6.72 · Semenyo 6.57 · Guéhi 6.29 · Gvardiol 5.44 · Rúben 4.43 · Donnarumma 4.18

Soft priors (DefCon/bonus/cards/saves) are zero here — side boards from the attack-roll trial helper do not carry expanding `*_mu` columns. Attack + CS/GC + appearance only.

Artifact: `reports/opta_player_roll_mancity_fulham.json`
