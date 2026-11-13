# Intensity trial on the 76-pixel card

Card: 76-pixel club / opponent / venue / 1X2. Baseline Poisson and fixture intensity only.
Native init is seed 0. Shifted init copies the body from a header four pixels taller,
with the training dropout stream matched to the native run.
Pass: painted gap ≤ 50 and both within 80 of 730 actual goals at the 20% cut.

| method | init | soft Σ | painted | exact | rows in (0,0.5) | goal chance by position |
|---|---|---:|---:|---:|---:|---|
| baseline | native | 923.8 | 907 | 170 | 205/548 | GKP 0.008, DEF 0.026, MID 0.069, FWD 0.146. |
| baseline | shifted | 1059.5 | 1046 | 191 | 187/548 | GKP 0.009, DEF 0.032, MID 0.079, FWD 0.161. |
| 3_fixture_intensity | native | 801.8 | 767 | 150 | 227/548 | GKP 0.005, DEF 0.020, MID 0.054, FWD 0.159. |
| 3_fixture_intensity | shifted | 788.8 | 762 | 145 | 215/548 | GKP 0.008, DEF 0.023, MID 0.051, FWD 0.151. |

Wall clock: 260 seconds on CPU.

