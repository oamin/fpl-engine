# Mean vs stable cascade path — per-GW residual (oracle minutes + autosub)

Protocol: `cascade_mean_path_gw_resid` · season **2025-26** · decision GW6 → horizon 8–19 · N=12 · seed=7

**Diagnostic.**
- **Semi-oracle pool** at GW6: actual minutes > 60 (n=126).
- **Minutes oracle** on boards: actual `m`, `σ_m=0` for the scored GW.
- **Actuals:** locked XI + FPL bench autosub + VC if captain blanks.
- Mean XI locked from priced t:t+1 MILP and held; path = strand-mean of held `xi_plus_captain_ft` lineups.
- Pred = oracle-minutes FD `score_xp_v2` on planned XI (+ captain).
- Residual = pred − actual.

- As-of rates/priors: GW `< 6` (historical rate baseline); minutes on boards are oracle per score GW
- Snapshots: [8, 10, 12, 14, 16, 18]
- Priced mean weeks: [8, 9]
- Wall: 89.4s

## Per GW

| GW | mean pred | mean actual | mean resid | mean autosubs | path pred | path actual | path resid | path−mean actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 85.0 | 52.0 | 33.0 | 0 | 61.2 | 66.8 | -5.6 | 14.8 |
| 9 | 87.6 | 56.0 | 31.6 | 0 | 62.7 | 55.8 | 6.9 | -0.2 |
| 10 | 79.1 | 69.0 | 10.1 | 0 | 62.2 | 63.3 | -1.1 | -5.7 |
| 11 | 76.4 | 36.0 | 40.4 | 0 | 60.9 | 40.5 | 20.4 | 4.5 |
| 12 | 71.8 | 37.0 | 34.8 | 2 | 61.6 | 37.2 | 24.4 | 0.2 |
| 13 | 69.2 | 38.0 | 31.2 | 2 | 53.3 | 29.4 | 23.9 | -8.6 |
| 14 | 70.8 | 49.0 | 21.8 | 2 | 52.9 | 51.7 | 1.3 | 2.7 |
| 15 | 66.1 | 42.0 | 24.1 | 2 | 55.8 | 57.5 | -1.7 | 15.5 |
| 16 | 73.0 | 66.0 | 7.0 | 1 | 61.0 | 60.5 | 0.5 | -5.5 |
| 17 | 76.3 | 71.0 | 5.3 | 2 | 58.3 | 67.2 | -8.9 | -3.8 |
| 18 | 62.4 | 47.0 | 15.4 | 2 | 42.3 | 48.1 | -5.8 | 1.1 |
| 19 | 65.0 | 42.0 | 23.0 | 1 | 40.5 | 46.4 | -5.9 | 4.4 |

## Horizon summary

- Mean residual (bias): **23.13** · MAE **23.13** · sum actual **605.0**
- Path residual (bias): **4.03** · MAE **8.88** · sum actual **624.3**
- Path − mean sum actual: **19.3**

## Artefacts

- `data/processed/cascade_mean_path_gw_resid/state.json`
- `data/predictions/2025-26/cascade_mean_path_gw_resid`
- `reports/cascade_mean_path_gw_resid.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid --season 2025-26 --n 12
```
