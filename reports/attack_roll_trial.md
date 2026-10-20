# Attack-roll trial — Arsenal v Everton (GW8 odds snapshot)

Generative short-horizon stack on one event:

1. `k ∼ M` (`match_outcome_roll` from fitted λ)
2. Each match goal: scorer ∼ `π` + `∅_G` mass `(1−α_G)`
3. If FPL scorer: assist ∼ `ρ` (scorer removed) + `∅_A` mass `(1−α_A)`

`π` = props-shape / baseline goal share (row 1). `ρ` = assist share only.

**Reproduce**

```bash
PYTHONPATH=. python3 -m src.markets.attack_roll_trial --n 12 --seed 0
```

**Inputs:** `data/scratch/gw08_odds_full.json` · rates from `player_gw_2026_27` (GW&lt;6).

**Note:** the 2026/27 cache lists some unexpected Arsenal names (Tzolis, Konsa, Bruno G.) — trial uses that squad as stored; matching is not the point of this smoke.

JSON: `reports/attack_roll_trial_arsenal_everton.json`.
