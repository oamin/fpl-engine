# Mean vs stable cascade path — per-GW residual (actual)

Protocol: `cascade_mean_path_gw_resid` · season **2025-26** · decision GW6 → horizon 8–19 · N=12 · seed=7

**Diagnostic.** Mean XI is locked from priced t:t+1 MILP and held. Path actual/pred are **strand means** of held `xi_plus_captain_ft` lineups (stable cascade path). Pred = FD centreline `score_xp_v2` on that GW's board for the planned XI (+ captain). Actual = Vaastav `total_points` on the planned XI (+ captain), no autosub. Residual = pred − actual.

- As-of freeze: GW `< 6` (rates use historical baseline)
- Snapshots: [8, 10, 12, 14, 16, 18]
- Priced mean weeks: [8, 9]
- Wall: 95.1s

## Per GW

| GW | mean pred | mean actual | mean resid | path pred | path actual | path resid | path−mean actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 70.8 | 58.0 | 12.8 | 60.2 | 67.7 | -7.5 | 9.7 |
| 9 | 73.4 | 42.0 | 31.4 | 60.6 | 47.6 | 13.0 | 5.6 |
| 10 | 68.6 | 75.0 | -6.4 | 62.0 | 63.9 | -1.9 | -11.1 |
| 11 | 69.2 | 34.0 | 35.2 | 59.8 | 42.2 | 17.6 | 8.2 |
| 12 | 73.7 | 23.0 | 50.7 | 62.2 | 31.5 | 30.7 | 8.5 |
| 13 | 70.7 | 37.0 | 33.7 | 53.9 | 27.2 | 26.6 | -9.8 |
| 14 | 73.3 | 48.0 | 25.3 | 59.7 | 42.9 | 16.8 | -5.1 |
| 15 | 71.5 | 74.0 | -2.5 | 60.4 | 62.6 | -2.2 | -11.4 |
| 16 | 70.4 | 81.0 | -10.6 | 59.8 | 56.2 | 3.6 | -24.8 |
| 17 | 72.8 | 72.0 | 0.8 | 61.9 | 65.5 | -3.6 | -6.5 |
| 18 | 73.0 | 40.0 | 33.0 | 57.2 | 45.5 | 11.7 | 5.5 |
| 19 | 72.8 | 31.0 | 41.8 | 55.8 | 33.9 | 21.9 | 2.9 |

## Horizon summary

- Mean residual (bias): **20.43** · MAE **23.67** · sum actual **615.0**
- Path residual (bias): **10.57** · MAE **13.08** · sum actual **586.7**
- Path − mean sum actual: **-28.3**

## Artefacts

- `data/processed/cascade_mean_path_gw_resid/state.json`
- `data/predictions/2025-26/cascade_mean_path_gw_resid`
- `reports/cascade_mean_path_gw_resid.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid --season 2025-26 --n 12
```
