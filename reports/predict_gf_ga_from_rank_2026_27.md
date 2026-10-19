# 2026/27 GF/GA from predicted rankings

Yes — with a caveat. Final table rank is a strong proxy for season GF/GA on closed seasons (ρ ≈ 0.84 / 0.78). Mapping is fit **only** on 2022/23–2025/26; the live 2026/27 holdout is not used to fit.

## Rank sources (most recent found)

| source | as-of | what |
|---|---|---|
| **Opta supercomputer** (press) | **2026-10-12** | predicted final positions + points |
| Betfair Exchange outrights → `E_rank` | 2026-10-08 | in-repo engine snapshot |

No newer full 1–20 table was found after 12 Oct in open press; club-level snippets exist through ~19 Oct (e.g. Chelsea still ~4th–5th in Opta).

## Mapping quality (closed seasons)

- OLS: `GF ≈ 79.9 -2.26·rank` (R²=0.67, MAE=7.29)
- OLS: `GA ≈ 36.3 +1.89·rank` (R²=0.6153, MAE=6.77)
- Leave-one-season-out MAE (rank-mean profile): GF **8.19**, GA **7.22** goals/season

Primary predictor below = **historical mean GF/GA at that finishing place** (interpolated for fractional Betfair E[rank]).

## Opta predicted table → GF / GA (2026-10-12)

| Rk | Club | Pred pts | Pred GF | Pred GA | Pred GD |
|---:|---|---:|---:|---:|---:|
| 1 | Man City | 78 | 87 | 34 | +53 |
| 2 | Arsenal | 77 | 81 | 35 | +46 |
| 3 | Brighton | 63 | 71 | 44 | +27 |
| 4 | Liverpool | 61 | 66 | 46 | +20 |
| 5 | Chelsea | 57 | 70 | 52 | +18 |
| 6 | Brentford | 54 | 66 | 55 | +11 |
| 7 | Newcastle | 53 | 59 | 50 | +8 |
| 8 | Everton | 53 | 61 | 56 | +5 |
| 9 | Man United | 52 | 58 | 54 | +3 |
| 10 | Aston Villa | 51 | 59 | 55 | +4 |
| 11 | Leeds | 51 | 49 | 54 | -5 |
| 12 | Hull | 46 | 46 | 51 | -6 |
| 13 | Ipswich | 46 | 44 | 55 | -11 |
| 14 | Crystal Palace | 45 | 48 | 58 | -10 |
| 15 | Bournemouth | 45 | 43 | 60 | -17 |
| 16 | Nott'm Forest | 44 | 49 | 63 | -14 |
| 17 | Tottenham | 44 | 49 | 62 | -13 |
| 18 | Sunderland | 42 | 46 | 74 | -29 |
| 19 | Coventry | 40 | 41 | 78 | -38 |
| 20 | Fulham | 38 | 31 | 83 | -52 |

## Betfair E[rank] → GF / GA (2026-10-08)

| Ord | Club | E[rank] | Pred GF | Pred GA | Pred GD |
|---:|---|---:|---:|---:|---:|
| 1 | Arsenal | 2.06 | 81 | 36 | +45 |
| 2 | Liverpool | 4.30 | 67 | 48 | +19 |
| 3 | Chelsea | 5.13 | 70 | 52 | +17 |
| 4 | Man Utd | 6.56 | 62 | 53 | +10 |
| 5 | Brighton | 7.51 | 60 | 54 | +7 |
| 6 | Brentford | 8.26 | 60 | 56 | +4 |
| 7 | Man City | 8.98 | 58 | 54 | +3 |
| 8 | Leeds | 10.48 | 54 | 54 | -0 |
| 9 | Aston Villa | 10.51 | 54 | 54 | -1 |
| 10 | Everton | 10.99 | 49 | 54 | -5 |
| 11 | Tottenham | 11.27 | 48 | 53 | -5 |
| 12 | Newcastle | 12.36 | 45 | 52 | -8 |
| 13 | Nottm Forest | 12.57 | 44 | 53 | -9 |
| 14 | Sunderland | 13.07 | 44 | 55 | -11 |
| 15 | Fulham | 13.65 | 46 | 57 | -11 |
| 16 | Crystal Palace | 13.96 | 48 | 58 | -10 |
| 17 | Ipswich | 14.83 | 44 | 60 | -16 |
| 18 | Hull | 15.75 | 48 | 62 | -15 |
| 19 | Coventry | 16.46 | 49 | 62 | -14 |
| 20 | Bournemouth | 18.50 | 43 | 76 | -33 |

## Historical GF/GA by finishing place (train mean)

| Rank | mean GF | mean GA | sd GF | sd GA |
|---:|---:|---:|---:|---:|
| 1 | 86.8 | 33.8 | 11.4 | 5.7 |
| 2 | 81.2 | 35.2 | 10.1 | 5.8 |
| 3 | 71.2 | 44.5 | 11.5 | 3.9 |
| 4 | 66.0 | 46.5 | 8.3 | 11.7 |
| 5 | 70.0 | 52.0 | 5.6 | 6.6 |
| 6 | 66.2 | 55.2 | 9.7 | 5.3 |
| 7 | 59.0 | 50.5 | 18.5 | 7.7 |
| 8 | 61.2 | 56.5 | 8.2 | 7.3 |
| 9 | 57.8 | 54.5 | 2.1 | 13.3 |
| 10 | 59.0 | 55.0 | 4.8 | 2.9 |
| 11 | 49.0 | 54.0 | 7.0 | 5.7 |
| 12 | 45.5 | 51.0 | 7.6 | 3.3 |
| 13 | 43.5 | 54.8 | 9.7 | 10.0 |
| 14 | 48.0 | 58.5 | 5.5 | 3.5 |
| 15 | 43.0 | 60.2 | 5.5 | 9.4 |
| 16 | 49.0 | 63.2 | 8.1 | 8.3 |
| 17 | 48.8 | 61.5 | 12.3 | 5.3 |
| 18 | 45.5 | 74.5 | 8.7 | 9.5 |
| 19 | 40.8 | 78.2 | 5.3 | 2.9 |
| 20 | 31.0 | 82.8 | 5.2 | 16.1 |

## Artefacts

- `data/processed/predict_gf_ga_from_rank_2026_27.csv`
- `reports/figures/predict_gf_ga_from_rank_2026_27.json`

Reproduce: `python3 scripts/predict_gf_ga_from_rank_2026_27.py`
