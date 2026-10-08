# Gameweek 6 stack audit (sensitivity)

Gemini ([stack audit](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): pre-deadline ablation of Exchange odds, coarse outright forecast, and contextual tags. This is input sensitivity, not realised performance.

Capture: `/workspace/data/predictions/2026-27/gw06/official_20261007T220914Z.csv`.
Betfair folder ready: **True** (`/workspace/data/predictions/2026-27/gw06/betfair_20261008`).

## Arms

| Arm | Config | Owned GW6 Σ | Rebuild−held GW6 | Chip | Line weeks |
| --- | --- | ---: | ---: | --- | --- |
| arm0_baseline | Odds API + rolling minutes + GW7 copy | 52.04 | 12.60 | wildcard | [6, 7] |
| arm1_tags | Odds API + contextual tags + GW7 copy | 54.86 | 12.56 | wildcard | [6, 7] |
| arm2_betfair | Betfair lines + tags + GW7 copy | 55.68 | 11.80 | wildcard | [6, 7] |
| arm3_full | Betfair lines + tags + outright forecast | 55.68 | 11.80 | wildcard | [6, 7] |

## Horizon outlooks (held XI)

| Arm | GW6 | GW7 | GW8 | Copy note |
| --- | ---: | ---: | ---: | --- |
| arm0_baseline | 54.79 | 56.70 | 56.70 | GW8 repeats GW7 and has no 1X2 of its own. |
| arm1_tags | 54.82 | 56.74 | 56.74 | GW8 repeats GW7 and has no 1X2 of its own. |
| arm2_betfair | 55.82 | 50.35 | 50.35 | GW8 repeats GW7 and has no 1X2 of its own. |
| arm3_full | 55.82 | 50.35 | 52.93 | (none — outrights filled unpriced weeks) |

## Deltas vs arm0 (not performance)

### arm1_tags

Owned GW6 Σ Δ = +2.822. Rebuild gap Δ = -0.036666666666668846. Chip = wildcard.
Copy note: GW8 repeats GW7 and has no 1X2 of its own.

| Player | Baseline | Arm | Δ |
| --- | ---: | ---: | ---: |
| van Ewijk | 0.234 | 3.016 | +2.782 |
| Calvert-Lewin | 2.753 | 2.770 | +0.017 |
| Cherki | 3.367 | 3.382 | +0.015 |
| Ødegaard | 5.025 | 5.012 | -0.013 |
| Rogers | 5.736 | 5.744 | +0.008 |
| Barnes | 3.815 | 3.823 | +0.008 |
| Shaw | 3.607 | 3.612 | +0.005 |

### arm2_betfair

Owned GW6 Σ Δ = +3.639. Rebuild gap Δ = -0.7941278703497687. Chip = wildcard.
Copy note: GW8 repeats GW7 and has no 1X2 of its own.

| Player | Baseline | Arm | Δ |
| --- | ---: | ---: | ---: |
| van Ewijk | 0.234 | 3.284 | +3.050 |
| Rogers | 5.736 | 6.222 | +0.486 |
| B.Fernandes | 5.633 | 6.023 | +0.390 |
| Barnes | 3.815 | 3.578 | -0.237 |
| Ødegaard | 5.025 | 4.904 | -0.121 |
| Haaland | 4.742 | 4.860 | +0.118 |
| Shaw | 3.607 | 3.516 | -0.091 |
| Guéhi | 3.553 | 3.638 | +0.085 |
| Lammens | 3.685 | 3.607 | -0.078 |
| Calafiori | 5.155 | 5.083 | -0.071 |
| Cherki | 3.367 | 3.436 | +0.068 |
| Calvert-Lewin | 2.753 | 2.796 | +0.042 |

### arm3_full

Owned GW6 Σ Δ = +3.639. Rebuild gap Δ = -0.7941278703497687. Chip = wildcard.

| Player | Baseline | Arm | Δ |
| --- | ---: | ---: | ---: |
| van Ewijk | 0.234 | 3.284 | +3.050 |
| Rogers | 5.736 | 6.222 | +0.486 |
| B.Fernandes | 5.633 | 6.023 | +0.390 |
| Barnes | 3.815 | 3.578 | -0.237 |
| Ødegaard | 5.025 | 4.904 | -0.121 |
| Haaland | 4.742 | 4.860 | +0.118 |
| Shaw | 3.607 | 3.516 | -0.091 |
| Guéhi | 3.553 | 3.638 | +0.085 |
| Lammens | 3.685 | 3.607 | -0.078 |
| Calafiori | 5.155 | 5.083 | -0.071 |
| Cherki | 3.367 | 3.436 | +0.068 |
| Calvert-Lewin | 2.753 | 2.796 | +0.042 |

## What this is not

- Not a claim that higher xP is better.
- Not a chip-rule test (the 114 wildcard lead is separate).
- Not a historical Exchange backtest (no books on file).
- Realised XI / Spearman wait until GW6 points are final.
