# Stage 27 — Game-engine play audit (locked xP v2)

Pure xP scorer (`FWD_LEVEL_CAL=False`, no horizon fade). Instrument the FT agent to see **where the game engine leaks points**, not whether μ ranks.

- Season: **2025-26**, GWs **4–38** (n=35)
- Defaults: H=3 (audit uses H=5), HOLD_EPS=1.25, HIT_COST=4
- FT total: **1942**
- Budget rebuild (no continuity): **2123** (gap -181)

## Season summary

| metric | value |
|---|---:|
| Total pts (cap − hits) | 1942 |
| Hit cost | −64 |
| Transfers | 48 (32 GWs) |
| Hold GWs | 2 |
| Hits taken | 16 |
| Mean ITB | £0.7m |
| Final squad SV | £84.4m |
| Autosubs fired | 40 |
| Blank XI slots left | 9 |

## Leak attribution (additive diagnostics)

| leak | pts | note |
|---|---:|---|
| Hit cost | 64 | paid −4s |
| Captain regret | 181 | best-in-XI actual − awarded C |
| Myopic hold regret | 42 | transfer GWs where hold scored more *this* GW |

These are **not** fully additive (captain regret ignores formation; hold regret is myopic). Use them to rank failure modes.

## Worst GWs (by gw_points)

| gw | pts | tx | hits | cap regret | blanks | held | captain |
|---:|---:|---:|---:|---:|---:|---:|---|
| 22 | 29 | 3 | 2 | 4 | 0 | 0 | Erling Haaland |
| 9 | 30 | 3 | 2 | 4 | 0 | 0 | Erling Haaland |
| 18 | 36 | 1 | 0 | 7 | 0 | 0 | Erling Haaland |
| 11 | 40 | 1 | 0 | 2 | 0 | 0 | Erling Haaland |
| 38 | 40 | 0 | 0 | 6 | 3 | 1 | Erling Haaland |
| 27 | 42 | 1 | 0 | 0 | 0 | 0 | Erling Haaland |
| 23 | 43 | 1 | 0 | 8 | 0 | 0 | Erling Haaland |
| 26 | 44 | 1 | 0 | 4 | 0 | 0 | Erling Haaland |

## Transfer behaviour

- Hold rate: **6%** of post-GW1 weeks
- Mean transfers/GW: **1.37**
- Hit rate: **16** hits over season

### Hit weeks

| gw | tx | hits | cost | Δ vs hold (myopic) |
|---:|---:|---:|---:|---:|
| 6 | 2 | 1 | 4 | +3.0 |
| 7 | 3 | 2 | 8 | -1.0 |
| 9 | 3 | 2 | 8 | +7.0 |
| 13 | 3 | 2 | 8 | -13.0 |
| 17 | 2 | 1 | 4 | -12.0 |
| 19 | 2 | 1 | 4 | +10.0 |
| 20 | 2 | 1 | 4 | -2.0 |
| 21 | 2 | 1 | 4 | -2.0 |
| 22 | 3 | 2 | 8 | +9.0 |
| 24 | 2 | 1 | 4 | -14.0 |
| 31 | 2 | 1 | 4 | -1.0 |
| 35 | 2 | 1 | 4 | -12.0 |

## Policy sensitivity

| setting | total | vs baseline | transfers | hits |
|---|---:|---:|---:|---:|
| baseline | 1942 | +0 | 48 | 16 |
| HOLD_EPS=0.5 | 1939 | -3 | 49 | 16 |
| HOLD_EPS=2.5 | 1918 | -24 | 49 | 17 |
| H=3 | 1923 | -19 | 34 | 1 |

## Gate read

- Continuity tax vs free budget rebuild: **+181** pts (rebuild 2123 − FT 1942)
- Largest measured leak bucket: **captain** (181 pts)

### Recommended next levers (from this audit)

1. **Captain model** — largest bucket (181). Top-μ C repeatedly regrets vs best-in-XI actual (Haaland lock weeks).
2. **Chips (WC/FH)** — continuity tax vs free rebuild is **181**; FT-only cannot close that.
3. **Hit discipline via H, not HOLD_EPS** — H=3 cut hits 16→1 (−19 pts overall); raising HOLD_EPS to 2.5 *hurt* (−24). Don’t crank EPS blindly.
4. ITB is fine (£0.7m); over-trading (hold only 6% of weeks) is the transfer-path smell.

## Outputs

- `data/processed/stage_27_decision_log.csv`
- `data/processed/stage_27_sensitivity.csv`
- `data/plots/stage_27_play_audit.png`
