# Stage 18 — Budgeted season climb

Each GW on the **eligible** (prior-3 max minutes ≥ 60) pool:

1. Build a **15-man squad** maximising Σ score under FPL rules (2/5/5/3, ≤3/club, Σ value ≤ **1000** = £100.0m)
2. Pick best legal **XI** from that squad by the same score
3. Captain = top score in XI; bank actual points

Fresh rebuild every GW (no transfer continuity). Primary scorer = stage-17 `ridge_global_starters`.

- GWs: **5–38** (n=34)

## Final standings (captain ×2)

| method | mode | total | mean/GW | vs ridge_budget |
|---|---|---:|---:|---:|
| ridge_global_starters_free | free | 2225 | 65.4 | +111 |
| xp_budget | budget | 2116 | 62.2 | +2 |
| ridge_global_starters_budget | budget | 2114 | 62.2 | +0 |
| xp_free | free | 2103 | 61.9 | -11 |
| price_free | free | 1963 | 57.7 | -151 |
| exp_points_free | free | 1849 | 54.4 | -265 |
| exp_points_budget | budget | 1817 | 53.4 | -297 |
| price_budget | budget | 1673 | 49.2 | -441 |

## Budget tax (same scorer)

- Ridge free − budget: **+111** (+3.26 / GW)
- Ridge budget vs exp budget: **+297**
- Ridge budget vs xp budget: **-2**

- Mean ridge budget squad value: **999** / 1000

## Gate verdict

**PARTIAL — beats exp_budget but not xp_budget (vs xp -2)**

## Plots

- `data/plots/season_climb_budget.png`

## Output

- `data/processed/season_climb_budget.csv`

## Read

- Large free−budget gap ⇒ unconstrained climb was premium-inflated.
- Ridge still useful under budget if it beats exp/xp on the budgeted gate.
