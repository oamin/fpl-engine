# Item 1b — Path-dependent FT / hits on multistage strands

Extends `xi_plus_captain` so each strand keeps a squad:

1. **First snapshot:** free rebuild (wildcard-like), FT bank → 1  
2. **Later snapshots:** maximise `XI + captain − 4·hits` over `k = 0 … ft+2` buys  
3. Skipped GWs between snapshots roll the FT bank with hold (`advance_ft(ft, 0)`)

Chips still off. Diagnostic only.

## Result (N=6, snapshot_every=2, seed=7 → 36 squads)

| Metric | Value |
| --- | ---: |
| Mean transfers / post-open snapshot | **2.13** |
| Mean hits / post-open snapshot | **0.67** |

| Player | £ | freq (FT path) | captain | stick_rate | freq (captain-only N=10) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saka | 9.6 | 0.67 | 0.50 | 0.86 | 0.85 |
| Haaland | 15.6 | **0.64** | 0.14 | **0.77** | 0.55 |
| Rogers | 7.8 | 0.72 | 0.03 | 0.86 | 0.73 |
| Barry | 5.7 | 0.69 | 0.00 | 0.87 | 0.53 |

Haaland still ~**#7**, but once bought he is sticky (77% kept across consecutive snapshots). FT friction raises his frequency vs independent captain rebuilds (0.64 vs 0.55) without making him automatic.

Cheap GK (Tzolakis) hits freq 1.0 / stick 1.0 — path mode locks budget enablers.

## Read

1. Transfers **matter for stability**, not for crowning a different elite set.  
2. Mean ~2 transfers + 0.7 hits / window is aggressive — horizon score swings still dominate the −4 hit cost.  
3. Next transformative levers (item 2+): minutes/role overlays, then match-band calibration.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp \
  --n 6 --snapshot-every 2 --seed 7 --objective xi_plus_captain_ft \
  --out data/predictions/2026-27/multistage_xi_captain_ft
```
