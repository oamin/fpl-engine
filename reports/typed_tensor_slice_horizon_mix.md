# Typed tensor — slice × horizon mix

Mix the sliced logistic spine with the horizon FOV card on remainder cells.
Train seasons 2022-23…2024-25; holdout 2025-26. Cut fractions 20/40/60/80%.

## Remainder holdout

| arm | log loss | Brier | on solid | on blank | n | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| slice | 0.1450 | 0.0388 | 0.145 | 0.042 | 80234 | slice on remainder cells |
| horizon | 0.1468 | 0.0392 | 0.135 | 0.040 | 80234 | epochs=22 |
| blend | 0.1468 | 0.0392 | 0.135 | 0.040 | 80234 | α=1.00 chosen on train |
| stack | 0.1464 | 0.0392 | 0.149 | 0.041 | 80234 | logit stack fit on train |
| residual | 0.1450 | 0.0388 | 0.144 | 0.041 | 80234 | epochs=12 |

## Blend grid (holdout; α = weight on horizon)

| α | log loss | Brier |
| ---: | ---: | ---: |
| 0.00 | 0.1450 | 0.0388 |
| 0.25 | 0.1447 | 0.0388 |
| 0.50 | 0.1448 | 0.0388 |
| 0.75 | 0.1455 | 0.0390 |
| 1.00 | 0.1468 | 0.0392 |

## Call

**residual** is hairline-best at **0.1450** (slice 0.1450, horizon 0.1468). Inconclusive — do not promote.
