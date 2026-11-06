# Mean vs stable cascade path — per-GW residual (oracle minutes + autosub)

Protocol: `cascade_mean_path_gw_resid` · season **2025-26** · decision GW6 → horizon 8–19 · N=12 · seed=7

**Diagnostic.**
- **Shares** on the full minutes-oracle side / as-of share roster (n_share=159).
- **Buy filter (semi-oracle)** at GW6: actual minutes > 60 (n_buy=126).
- **Minutes oracle** on residual/mean boards: actual `m`, `σ_m=0` for the scored GW (no GW(t+n) foresight in path generative minutes).
- **Actuals:** locked XI + FPL bench autosub + VC if captain blanks.
- Mean XI locked from priced t:t+1 MILP and held; path = strand-mean of held `xi_plus_captain_ft` lineups.
- Pred = oracle-minutes FD `score_xp_v2` on planned XI (+ captain).
- Residual = pred − actual.

- As-of rates/priors: GW `< 6` (historical rate baseline); minutes on boards are oracle per score GW
- Snapshots: [8, 10, 12, 14, 16, 18]
- Priced mean weeks: [8, 9]
- Wall: 95.4s

## Per GW

| GW | mean pred | mean actual | mean resid | mean autosubs | path pred | path actual | path resid | path−mean actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 75.3 | 78.0 | -2.7 | 0 | 61.3 | 74.0 | -12.7 | -4.0 |
| 9 | 71.5 | 49.0 | 22.5 | 0 | 59.9 | 53.4 | 6.5 | 4.4 |
| 10 | 69.4 | 63.0 | 6.4 | 0 | 57.5 | 66.6 | -9.1 | 3.6 |
| 11 | 68.4 | 34.0 | 34.4 | 0 | 57.5 | 41.7 | 15.8 | 7.7 |
| 12 | 65.5 | 35.0 | 30.5 | 1 | 55.1 | 36.8 | 18.3 | 1.8 |
| 13 | 70.1 | 32.0 | 38.1 | 0 | 58.3 | 28.4 | 29.8 | -3.6 |
| 14 | 61.2 | 55.0 | 6.2 | 2 | 53.1 | 50.6 | 2.5 | -4.4 |
| 15 | 61.8 | 61.0 | 0.8 | 0 | 56.2 | 64.1 | -7.9 | 3.1 |
| 16 | 65.1 | 76.0 | -10.9 | 0 | 56.6 | 58.7 | -2.1 | -17.3 |
| 17 | 68.9 | 68.0 | 0.9 | 0 | 60.5 | 69.0 | -8.5 | 1.0 |
| 18 | 59.5 | 45.0 | 14.5 | 2 | 44.0 | 54.8 | -10.8 | 9.8 |
| 19 | 62.2 | 51.0 | 11.2 | 1 | 39.3 | 48.5 | -9.2 | -2.5 |

## Horizon summary

- Mean residual (bias): **12.65** · MAE **14.92** · sum actual **647.0**
- Path residual (bias): **1.06** · MAE **11.10** · sum actual **646.4**
- Path − mean sum actual: **-0.6**

## Artefacts

- `data/processed/cascade_mean_path_gw_resid/state.json`
- `data/predictions/2025-26/cascade_mean_path_gw_resid`
- `reports/cascade_mean_path_gw_resid.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid --season 2025-26 --n 12
```
