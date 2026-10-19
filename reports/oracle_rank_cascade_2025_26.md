# Oracle-rank cascade diagnostic — 2025/26

Retrospective ceiling test. Final 2025/26 table ranks are treated as **predicted** ranks. Rank → GF/GA OLS is fit on **2022/23–2024/25 only**. Match pots use Dixon–Coles from those season rates; player xP uses deadline shares × `xp_on_pot` (cascade path). Not a live method. 2026/27 unused.

## Map fit (prior seasons)

- Files: E0_2223.csv, E0_2324.csv, E0_2425.csv (exclude `E0_2526.csv`)
- GF ≈ 83.17 + (-2.452)·rank  (MAE 7.73)
- GA ≈ 35.78 + (2.061)·rank  (MAE 6.79)
- League avg GF (prior profile): 57.42

Promoted clubs (prior-rank control): {'burnley': 18.0, 'leeds': 19.0, 'sunderland': 20.0}

## Club-season GF/GA (predicted from ranks vs realised)

| variant | MAE GF | MAE GA | corr GF | corr GA |
| --- | ---: | ---: | ---: | ---: |
| oracle | 7.08 | 7.75 | 0.849 | 0.792 |
| prior_rank | 10.23 | 8.79 | 0.660 | 0.574 |
| flat | 10.32 | 8.74 | 0.000 | 0.000 |

## Match λ vs actual goals

| variant | n fixtures | MSE | MAE | RMSE | band coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 380 | 1.226 | 0.903 | 1.107 | 0.263 |
| prior_rank | 380 | 1.296 | 0.933 | 1.138 | 0.249 |
| flat | 380 | 1.290 | 0.959 | 1.136 | 0.136 |

MC strands (N=40): oracle MAE(strand mean)=0.902 vs MAE(centreline λ)=0.903.

## Eligible player-GW cascade xP (GW5–38, n_prior≥3, xmi≥45)

| variant | n | mean log | MAE | bias | mean xP |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 7444 | -2.53427 | 2.314 | +0.168 | 3.171 |
| prior_rank | 7444 | -2.54510 | 2.343 | +0.162 | 3.165 |
| flat | 7444 | -2.54554 | 2.342 | +0.104 | 3.108 |
| market_shin_score_xp | 7444 | -2.53811 | 2.374 | +0.324 | 3.328 |

## Viability

**PASS** vs flat / prior-rank controls (match MSE + player log).
Oracle − market log-score: +0.00384 (reference only; market uses fixture odds).

## Outputs

- `data/processed/oracle_rank_cascade_2025_26.json`
- `data/processed/oracle_rank_cascade_club_errors.csv`

Reproduce: `python -m src.eval.oracle_rank_cascade`.
