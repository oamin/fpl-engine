# Stage 22 add-on — FPL autosubs + vice-captain

GW scoring now mirrors FPL:

1. Pick XI by μ; remaining 4 = bench (outfield ordered by μ)
2. **Autosubs:** 0-minute starters replaced from bench if sub played and formation stays legal (GK↔GK only)
3. **Captain / VC:** captain = top μ on intended XI; if captain blanks, VC gets the double

## ridge_h3_ft GW5–38

| variant | total | vs prior 1872 |
|---|---:|---:|
| prior (no autosub, no VC) | 1872 | — |
| + VC only (no autosub) | 1917 | +45 |
| + VC + autosubs | 1996 | +124 |

- Autosubs applied: **20**
- Intended XI blanks: **22** → final blanks **2**
- Points recovered by autosubs alone: **+79**

Wired into `bank_squad_gw` in `season_climb.py`; used by FT / budget / hybrid / stage22–23 scorers.
