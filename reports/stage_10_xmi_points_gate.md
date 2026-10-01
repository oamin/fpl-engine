# Stage 10 — Gate: xMi → next-match points

Does **roll3 xMi** alone forecast next-GW `total_points`? Compared to roll3 prior points and xMi×prior points-per-minute.

- Raw rows: **29338**
- Backtest: prior GWs ≥ 3
- OLS xMi→pts: intercept=0.2067, slope=0.0374

## Headline (scope=ALL)

### Filter: `ALL`

| pair | n | MAE | R² | corr | decile_corr | Δ top−bot | mean y |
|---|---:|---:|---:|---:|---:|---:|---:|
| xMi → points (raw mins scale) | 26819 | 24.232 | -303.684 | 0.547 | 0.983 | 3.133 | 1.152 |
| xMi → points (OLS calibrated) | 26819 | 1.046 | 0.300 | 0.547 | 0.983 | 3.133 | 1.152 |
| xMi×prior_ppm → points | 26819 | 1.002 | 0.260 | 0.540 | 0.970 | 3.518 | 1.152 |
| xMi×ppm OLS → points | 26819 | 1.064 | 0.292 | 0.540 | 0.970 | 3.518 | 1.152 |
| roll3 points → points | 26819 | 1.043 | 0.160 | 0.491 | 0.887 | 3.293 | 1.152 |
| roll3 start rate → points | 26819 | 0.981 | 0.013 | 0.523 | nan | 2.841 | 1.152 |

### Filter: `mins>0`

| pair | n | MAE | R² | corr | decile_corr | Δ top−bot | mean y |
|---|---:|---:|---:|---:|---:|---:|---:|
| xMi → points (raw mins scale) | 10327 | 53.700 | -435.430 | 0.227 | 0.989 | 1.987 | 2.992 |
| xMi → points (OLS calibrated) | 10327 | 2.040 | -0.031 | 0.227 | 0.989 | 1.987 | 2.992 |
| xMi×prior_ppm → points | 10327 | 2.178 | -0.074 | 0.238 | 0.983 | 2.321 | 2.992 |
| xMi×ppm OLS → points | 10327 | 2.018 | -0.045 | 0.238 | 0.983 | 2.321 | 2.992 |
| roll3 points → points | 10327 | 2.299 | -0.227 | 0.188 | 0.907 | 2.004 | 2.992 |
| roll3 start rate → points | 10327 | 2.443 | -0.594 | 0.212 | nan | 1.320 | 2.992 |

### Filter: `mins>=60`

| pair | n | MAE | R² | corr | decile_corr | Δ top−bot | mean y |
|---|---:|---:|---:|---:|---:|---:|---:|
| xMi → points (raw mins scale) | 7043 | 64.450 | -485.993 | 0.020 | 0.585 | 0.229 | 3.817 |
| xMi → points (OLS calibrated) | 7043 | 2.436 | -0.203 | 0.020 | 0.585 | 0.229 | 3.817 |
| xMi×prior_ppm → points | 7043 | 2.551 | -0.215 | 0.086 | 0.867 | 0.864 | 3.817 |
| xMi×ppm OLS → points | 7043 | 2.426 | -0.221 | 0.086 | 0.867 | 0.864 | 3.817 |
| roll3 points → points | 7043 | 2.735 | -0.394 | 0.055 | 0.779 | 0.800 | 3.817 |
| roll3 start rate → points | 7043 | 3.115 | -0.943 | 0.014 | nan | 0.099 | 3.817 |

### Filter: `xmi>=45`

| pair | n | MAE | R² | corr | decile_corr | Δ top−bot | mean y |
|---|---:|---:|---:|---:|---:|---:|---:|
| xMi → points (raw mins scale) | 7547 | 73.766 | -573.193 | 0.166 | 0.950 | 1.326 | 3.008 |
| xMi → points (OLS calibrated) | 7547 | 2.339 | 0.027 | 0.166 | 0.950 | 1.326 | 3.008 |
| xMi×prior_ppm → points | 7547 | 2.513 | -0.048 | 0.180 | 0.971 | 1.857 | 3.008 |
| xMi×ppm OLS → points | 7547 | 2.288 | 0.015 | 0.180 | 0.971 | 1.857 | 3.008 |
| roll3 points → points | 7547 | 2.622 | -0.229 | 0.126 | 0.897 | 1.486 | 3.008 |
| roll3 start rate → points | 7547 | 2.414 | -0.448 | 0.119 | nan | 0.842 | 3.008 |

## Minutes reference (sanity)

- xMi → minutes: R²=0.601, corr=0.787, MAE=11.42

## By position — xMi OLS → points (ALL rows)

| pos | n | MAE | R² | corr | Δ top−bot |
|---|---:|---:|---:|---:|---:|
| GKP | 3092 | 0.743 | 0.422 | 0.662 | 3.009 |
| DEF | 8791 | 1.225 | 0.256 | 0.509 | 3.182 |
| MID | 11983 | 0.987 | 0.312 | 0.560 | 3.501 |
| FWD | 2953 | 1.070 | 0.288 | 0.550 | 3.702 |

## Plots

- `data/plots/xmi_points_gate.png`
- `data/plots/xmi_points_gate_by_pos.png`

## Output

- `data/processed/xmi_points_gate.csv`

## Gate read

- If xMi→points corr ≪ xMi→minutes, minutes forecast well but **points ceiling is thin**.
- If roll3 points ≫ xMi for points, historic scoring rate dominates appearance.
- If both weak once mins≥60, appearance is the whole story and event noise remains.
