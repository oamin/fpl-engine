# Oracle-rank cascade diagnostic — 2025/26

Retrospective ceiling test. Final 2025/26 table ranks are treated as **predicted** ranks. Rank → GF/GA OLS is fit on **2022/23–2024/25 only**. Match λ from those season rates; player xP = Dixon–Coles `M` → `score_team_side` (`score_xp_v2`). Not a live method. 2026/27 unused.

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

## Eligible player-GW cascade `score_xp_v2` (GW5–38, n_prior≥3, minutes>60)

| variant | n | mean log | MAE | bias | mean xP |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 6737 | -2.55376 | 2.354 | -0.031 | 3.796 |
| prior_rank | 6737 | -2.56720 | 2.388 | -0.044 | 3.783 |
| flat | 6737 | -2.57476 | 2.414 | -0.094 | 3.733 |
| market_fd_score_xp_v2 | 6856 | -2.57248 | 2.364 | -0.099 | 3.719 |

## Viability

**PASS** vs flat / prior-rank controls (match MSE + player log).
Oracle − market `score_xp_v2` log-score: +0.01873 (reference only; market uses FD AH+OU pots).

## Outputs

- `data/processed/oracle_rank_cascade_2025_26.json`
- `data/processed/oracle_rank_cascade_club_errors.csv`

Reproduce: `python -m src.eval.oracle_rank_cascade`.
