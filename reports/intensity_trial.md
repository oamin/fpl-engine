# Intensity allocation trial

Card: 72-pixel odds / venue / opponent. Goals only change head. Assists stay four-way.
Native init is seed 0. Shifted init copies the body from a 76-pixel header draw.
Pass: painted gap ≤ 50, both within 80 of 730 actual goals, keepers below defenders,
and GKP < DEF < MID < FWD on chance. Proposal 2 (presence separate from height) is
built into the row-intensity and STE runs.

| method | init | soft Σ | painted | exact | rows in (0,0.5) | goal chance by position | gate |
|---|---|---:|---:|---:|---:|---|---|
| baseline (Poisson cell) | native | 803.1 | 779 | 149 | 228/548 | GKP 0.004, DEF 0.022, MID 0.060, FWD 0.132 | init gap 9; hierarchy ok; volume ok |
| baseline (Poisson cell) | shifted | 783.9 | 770 | 127 | 209/548 | GKP 0.014, DEF 0.022, MID 0.056, FWD 0.132 | |
| 1 row intensity (+2) | native | 613.5 | 589 | 98 | 203/548 | GKP 0.047, DEF 0.022, MID 0.044, FWD 0.065 | FAIL: under volume; GKP > DEF |
| 1 row intensity (+2) | shifted | 626.8 | 610 | 99 | 199/548 | GKP 0.049, DEF 0.022, MID 0.045, FWD 0.067 | init gap 21 |
| 3 fixture intensity | native | 793.3 | 770 | 148 | 219/548 | GKP 0.005, DEF 0.020, MID 0.061, FWD 0.128 | PASS: gap 3; volume near 730; hierarchy ok |
| 3 fixture intensity | shifted | 796.4 | 767 | 148 | 221/548 | GKP 0.005, DEF 0.020, MID 0.059, FWD 0.136 | |
| 1+4 row intensity + STE | native | 645.8 | 628 | 103 | 191/548 | GKP 0.053, DEF 0.019, MID 0.048, FWD 0.071 | FAIL: under volume; GKP > DEF |
| 1+4 row intensity + STE | shifted | 641.6 | 628 | 103 | 178/548 | GKP 0.054, DEF 0.019, MID 0.047, FWD 0.073 | init gap 0 |

Wall clock: 525 seconds on CPU.

Winner under the dual-init gate: proposal 3, fixture intensity then share down the squad.

