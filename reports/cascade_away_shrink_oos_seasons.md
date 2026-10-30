# OOS club-λ screen: away_shrink=0.768 on prior seasons

Candidate **s=0.768** (from 2025-26 XI resid crossing) vs **s=1.0** · GW8–38 · N=40 · wall 73.4s

**Verdict (2026-10-27):** park fixed `s`; live default **`AWAY_SHRINK=1.0`**.
Keep away-only shrink formula and **no** 0.20–3.50 λ clip.

## Why these seasons

- `s=0.768` was read from **2025-26** XI resid G/CS → that season is selection-contaminated for XI.
- Shared L3 XI seed/path exists only for 2025-26.
- Club λ on **2022-23 / 2023-24 / 2024-25** is the clean OOS check.

Promote (per season vs that season's s=1.0): away |bias| ↓ ≥0.1, home |bias| ↑ ≤0.05, MAE ↑ ≤0.02.

## Per-season holdout (Opta λ − actual)

| season | s=1.0 bias A | s=1.0 bias H | s=0.768 bias A | s=0.768 bias H | MAE Δ | promote |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2022-23 | +0.218 | -0.067 | -0.129 | -0.068 | -0.019 | FAIL |
| 2023-24 | -0.056 | -0.421 | -0.385 | -0.421 | +0.013 | FAIL |
| 2024-25 | +0.068 | -0.039 | -0.276 | -0.039 | -0.006 | FAIL |

**Per-season PASS count:** 0/3

## Pooled 2022-23…2024-25

- baseline bias A/H/MAE = +0.077 / -0.174 / 0.957
- s=0.768 bias A/H/MAE = -0.263 / -0.174 / 0.953
- promote: **FAIL** (away_improve=-0.186, home_worsen=+0.000, mae_worsen=-0.004)

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_away_shrink_oos_seasons
```

