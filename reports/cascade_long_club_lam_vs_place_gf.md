# Cascade_long club Σλ vs place GF

Pre-registered **2026-10-28** · protocol `cascade_long_club_lam_vs_place_gf` · wall 14.8s

## Verdict

- Pooled mean **Σλ − place GF ≈ +0.10** (~0 → design converges).
- Pooled mean **Σλ − actual ≈ +0.88** (can be hot when the season scores below the LOSO place prior).
- Venue: home ~calibrated, away high — **misallocation inside the season total**, not escape from place GF.
- XI xP(G) overprint is **not** explained by club Σλ drifting off place GF.

## Per season (club means)

| season | Σλ | place GF | actual | λ−place | λ−act | bias H | bias A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 57.0 | 56.8 | 52.8 | +0.20 | +4.17 | -0.153 | +0.271 |
| 2023-24 | 54.1 | 54.1 | 61.5 | +0.07 | -7.37 | -0.369 | -0.062 |
| 2024-25 | 56.3 | 56.2 | 55.4 | +0.06 | +0.91 | -0.030 | +0.056 |
| 2025-26 | 57.5 | 57.4 | 51.7 | +0.09 | +5.81 | -0.007 | +0.286 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_club_lam_vs_place_gf
```

