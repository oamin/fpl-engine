# Stage 19 — Transfer-constrained season climb

FPL rules (chips off):

- **First scored GW:** free 15 under £100.0m (wildcard-like)
- **Thereafter:** 1 FT / GW, stack to **5**; extras **−4** each (explore 0–2 hits)
- **Sell price:** purchase + ⌊rise/2⌋; full fall to current (`value`)
- **Transfer policy:** enumerate hold / swaps; V = Σ γ^h XI_score (H=3, γ=0.9) − 4·hits; hold unless ΔV ≥ 1.25
- Scores frozen from decision GW; future blanks (missing Vaastav row) → 0
- Buy pool = eligible ∪ currently owned; XI from squad by score

- GWs: **5–38** (n=34)

## Final standings (captain ×2 − hits)

| method | mode | total | mean/GW | vs ridge_ft |
|---|---|---:|---:|---:|
| xp_budget | budget | 2116 | 62.2 | +370 |
| ridge_global_starters_budget | budget | 2114 | 62.2 | +368 |
| xp_ft | ft | 1848 | 54.4 | +102 |
| exp_points_budget | budget | 1817 | 53.4 | +71 |
| ridge_global_starters_ft | ft | 1746 | 51.4 | +0 |
| exp_points_ft | ft | 1644 | 48.4 | -102 |

## Transfer friction

- Ridge FT mean transfers/GW: **0.97**
- Ridge FT hold weeks: **4** / 34
- Ridge FT max ft_before seen: **2**
- Ridge FT total hits: **1** (−4 pts)
- Ridge FT vs budget rebuild: **-368**
- Ridge FT vs xp FT: **-102**
- Ridge FT vs exp FT: **+102**

## Gate verdict

**PARTIAL — beats exp_ft but not xp_ft (vs xp -102)**

## Plots

- `data/plots/season_climb_ft.png`

## Output

- `data/processed/season_climb_ft.csv`

## Read

- FT stack + hits is the real decision surface; budget rebuild was an upper bound.
- Large FT≪budget gap ⇒ value is locked in transfer timing, not just ranking.
- Policy uses XI-horizon V (not Σ15); hold unless ΔV ≥ 1.25.
