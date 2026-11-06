# Deadline-causal minutes vs open-loop path plan

Protocol: `cascade_deadline_causal_minutes` · season **2025-26** · decision GW6 → 8–19 · N=12 · seed=7

FT banking is insurance for unexpected minutes/rotations. Control freezes minutes for the whole horizon (open-loop joint MILP). Candidate reveals oracle minutes only at each GW deadline and replans (as-of minutes for later weeks).

| arm | minutes / planner |
| --- | --- |
| control | frozen as-of (+ GW-t buy overlay); `open_loop` joint MILP |
| candidate | oracle @ deadline GW only; as-of for later; `deadline_causal` rolling replan |

Both arms use terminal FT PE ν=4. Wall: 216.3s

## Transfer behaviour (mean over strands; excl. opening free rebuild)

| arm | transfers | free used | paid | hit pts | mean terminal ft |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 26.58 | 10.92 | 15.67 | 62.7 | 1.08 |
| candidate | 54.33 | 11.00 | 43.33 | 173.3 | 1.00 |
| Δ (cand−ctrl) | +27.75 |  | +27.67 | +110.7 | -0.08 |

## Path residual vs actual

| arm | sum actual | resid bias | MAE |
| --- | ---: | ---: | ---: |
| control | 630.0 | -2.87 | 8.27 |
| candidate | 641.8 | +2.70 | 6.50 |
| Δ | +11.8 | +5.58 |  |

## Decision

**`keep_watching_causal_moves`** — deadline-causal minutes make the path react
(much higher paid transfers) and lift path actual (~+12), but terminal FT bank
does **not** rise. Hit-value PE alone is not enough to prefer banking through
minutes shocks; diagnostic only (not a four-season promote).

## Artefacts

- `data/processed/cascade_deadline_causal_minutes/state.json`
- `data/predictions/2025-26/cascade_deadline_causal_minutes`
- `reports/cascade_deadline_causal_minutes.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_deadline_causal_minutes --season 2025-26 --n 12
```
