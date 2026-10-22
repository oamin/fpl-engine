# Path bench-cover weight (α) — rejected

Protocol: `cascade_path_bench_weight` · season **2025-26** · N=12

Tried `V = Σ [α·squad_s + (1-α)·XI_s + captain_s] − hits [+ U(f)]`.

| α | Δpaid | Δhit pts | Δ actual |
| ---: | ---: | ---: | ---: |
| 0.10 | +0.25 | +1 | +4.3 |
| 0.50 | +5.75 | +23 | +2.0 |

## Decision

**`reject_bench_alpha`** — not the intended bench/rotation design. α removed
from the path MILP objective; engine is again XI + captain only (− hits [+ PE]).
