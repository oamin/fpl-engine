# Fixture covariance demo (M → minutes → shares → xP)

Toy home favourites with a competing FB pair (sparse minutes) and volatile goal/assist Dirichlet shares. Scoreline sampled from Dixon–Coles M.

- Scenarios: **500**
- Mean |draw μ − point μ|: **0.361**
- Scoreline mean: **1.66–1.11**

## Induced dependence

- Corr home M1–M2 xP: **+0.595** (shared attack pot)
- Corr home M1 vs away GK xP: **-0.717** (CS/GC)
- Corr competing FB **minutes**: **-0.988** (sparse coupler)
- Corr competing FB **xP**: **-0.282**
- Home FWD mean xP | home blank: **2.27**; | home ≥2 goals: **8.19**

## Player summary

| id | side | point μ | draw μ | draw SD | mean m |
| --- | --- | ---: | ---: | ---: | ---: |
| H_GK | home | 4.35 | 4.28 | 2.21 | 86.9 |
| H_D1 | home | 6.18 | 4.93 | 3.23 | 67.8 |
| H_D2 | home | 1.15 | 1.80 | 1.95 | 35.1 |
| H_M1 | home | 7.12 | 6.91 | 3.59 | 83.2 |
| H_M2 | home | 5.00 | 4.55 | 2.46 | 68.5 |
| H_F | home | 5.93 | 5.86 | 3.01 | 79.4 |
| A_GK | away | 3.40 | 3.42 | 1.93 | 86.9 |
| A_D1 | away | 4.81 | 3.90 | 2.68 | 69.1 |
| A_D2 | away | 0.74 | 1.21 | 1.48 | 33.8 |
| A_M1 | away | 5.45 | 5.34 | 2.78 | 83.6 |
| A_M2 | away | 4.03 | 3.92 | 2.05 | 70.0 |
| A_F | away | 4.66 | 4.67 | 2.47 | 78.7 |

## Verdict

Sparse FB minutes corr -0.99 (ok). Attack–opp GK -0.72 (ok). Home mids +0.60 (ok). Fixture path induces the right qualitative couplings without a dense Σ.

JSON: `data/processed/fixture_covariance_demo.json`

