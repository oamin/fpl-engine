# Path bench-cover weight (α) trial

Protocol: `cascade_path_bench_weight` · season **2025-26** · decision GW6 → 8–19 · N=12 · seed=7

Bench has zero weight under legacy XI+cap. Candidate credits cover:

`V = Σ [α·squad_s + (1-α)·XI_s + captain_s] − hits [+ U(f)]` with **α=0.1** (locked a priori; not fitted on path actual).

| arm | objective |
| --- | --- |
| control | `α=0` (XI + captain only) |
| candidate | `α=0.1` |

Open-loop joint path · FT PE ν=4. Wall: 114.9s

## Transfer behaviour (mean over strands; excl. opening free rebuild)

| arm | transfers | free used | paid | hit pts | mean terminal ft |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 26.58 | 10.92 | 15.67 | 62.7 | 1.08 |
| candidate | 26.83 | 10.92 | 15.92 | 63.7 | 1.08 |
| Δ (cand−ctrl) | +0.25 |  | +0.25 | +1.0 | +0.00 |

## Path residual vs actual

| arm | sum actual | resid bias | MAE |
| --- | ---: | ---: | ---: |
| control | 630.0 | -2.87 | 8.27 |
| candidate | 634.3 | -3.07 | 8.41 |
| Δ | +4.3 | -0.20 |  |

## Decision

**`keep_watching_bench_alpha_moves`** — α=0.10 barely moves the schedule
(Δpaid +0.25) and lifts path actual ~+4. Does **not** cut hit churn on this
open-loop board. Diagnostic only (not a four-season promote); do not default α on yet.

## Artefacts

- `data/processed/cascade_path_bench_weight/state.json`
- `data/predictions/2025-26/cascade_path_bench_weight`
- `reports/cascade_path_bench_weight.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_path_bench_weight --season 2025-26 --n 12 --alpha 0.1
```
