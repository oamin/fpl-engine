# Stage 3 — Team pots vs 1X2 / OU / AH

Go thresholds: |decile corr| ≥ 0.55 **or** match R² ≥ 0.02.

| pos | feature | n | slope | R² | corr | decile_corr | go |
|---|---|---:|---:|---:|---:|---:|:---:|
| GKP | defend_threat | 760 | -2.532 | 0.030 | -0.172 | -0.860 | Y |
| GKP | p_over | 760 | -1.679 | 0.002 | -0.046 | -0.507 | N |
| GKP | ah_line_own | 758 | -0.607 | 0.033 | -0.181 | -0.919 | Y |
| GKP | p_ah_cover | 760 | -5.769 | 0.003 | -0.054 | -0.352 | N |
| DEF | defend_threat | 760 | -15.136 | 0.063 | -0.252 | -0.918 | Y |
| DEF | p_over | 760 | -10.619 | 0.005 | -0.071 | -0.637 | Y |
| DEF | ah_line_own | 758 | -3.418 | 0.062 | -0.249 | -0.970 | Y |
| DEF | p_ah_cover | 760 | -21.517 | 0.002 | -0.049 | -0.491 | N |
| MID | attack_strength | 760 | 17.892 | 0.146 | 0.382 | 0.969 | Y |
| MID | p_over | 760 | 5.798 | 0.002 | 0.050 | 0.582 | Y |
| MID | ah_line_own | 758 | -4.019 | 0.142 | -0.376 | -0.979 | Y |
| MID | p_ah_cover | 760 | -14.663 | 0.002 | -0.043 | -0.331 | N |
| FWD | attack_strength | 748 | 1.100 | 0.002 | 0.046 | 0.354 | N |
| FWD | p_over | 748 | 0.774 | 0.000 | 0.013 | 0.003 | N |
| FWD | ah_line_own | 746 | -0.279 | 0.003 | -0.050 | -0.519 | N |
| FWD | p_ah_cover | 748 | -10.797 | 0.004 | -0.061 | -0.576 | Y |

## Position go/no-go (any feature passes)

- **GKP**: GO
- **DEF**: GO
- **MID**: GO
- **FWD**: GO

## Plots

- `data/plots/stage_3_team_market.png` — match-level scatter
- `data/plots/stage_3_gate_deciles.png` — **gate check** (decile means)
- `data/plots/stage_3_gate_combined.png` — scatter vs decile by position

Output: `data/processed/team_market_eval.csv`
