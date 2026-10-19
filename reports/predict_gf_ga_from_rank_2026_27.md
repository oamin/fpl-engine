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

Headline Pred GF/GA use the **OLS** map (monotonic in rank). Rank-mean profile is also stored (can be non-monotonic with only four seasons).

Note: Betfair `E_rank` prices Man City mid-table because of elevated relegation risk on the Exchange; Opta assumes no sporting sanction.

## Opta predicted table → GF / GA (2026-10-12)

| Rk | Club | Pred pts | Pred GF | Pred GA | Pred GD |
|---:|---|---:|---:|---:|---:|
| 1 | Man City | 78 | 78 | 38 | +39 |
| 2 | Arsenal | 77 | 75 | 40 | +35 |
| 3 | Brighton | 63 | 73 | 42 | +31 |
| 4 | Liverpool | 61 | 71 | 44 | +27 |
| 5 | Chelsea | 57 | 69 | 46 | +23 |
| 6 | Brentford | 54 | 66 | 48 | +19 |
| 7 | Newcastle | 53 | 64 | 50 | +14 |
| 8 | Everton | 53 | 62 | 51 | +10 |
| 9 | Man United | 52 | 60 | 53 | +6 |
| 10 | Aston Villa | 51 | 57 | 55 | +2 |
| 11 | Leeds | 51 | 55 | 57 | -2 |
| 12 | Hull | 46 | 53 | 59 | -6 |
| 13 | Ipswich | 46 | 50 | 61 | -10 |
| 14 | Crystal Palace | 45 | 48 | 63 | -14 |
| 15 | Bournemouth | 45 | 46 | 65 | -19 |
| 16 | Nott'm Forest | 44 | 44 | 66 | -23 |
| 17 | Tottenham | 44 | 41 | 68 | -27 |
| 18 | Sunderland | 42 | 39 | 70 | -31 |
| 19 | Coventry | 40 | 37 | 72 | -35 |
| 20 | Fulham | 38 | 35 | 74 | -39 |

## Betfair E[rank] → GF / GA (2026-10-08)

| Ord | Club | E[rank] | Pred GF | Pred GA | Pred GD |
|---:|---|---:|---:|---:|---:|
| 1 | Arsenal | 2.06 | 75 | 40 | +35 |
| 2 | Liverpool | 4.30 | 70 | 44 | +26 |
| 3 | Chelsea | 5.13 | 68 | 46 | +22 |
| 4 | Man Utd | 6.56 | 65 | 49 | +16 |
| 5 | Brighton | 7.51 | 63 | 50 | +12 |
| 6 | Brentford | 8.26 | 61 | 52 | +9 |
| 7 | Man City | 8.98 | 60 | 53 | +6 |
| 8 | Leeds | 10.48 | 56 | 56 | +0 |
| 9 | Aston Villa | 10.51 | 56 | 56 | -0 |
| 10 | Everton | 10.99 | 55 | 57 | -2 |
| 11 | Tottenham | 11.27 | 54 | 58 | -3 |
| 12 | Newcastle | 12.36 | 52 | 60 | -8 |
| 13 | Nottm Forest | 12.57 | 51 | 60 | -9 |
| 14 | Sunderland | 13.07 | 50 | 61 | -11 |
| 15 | Fulham | 13.65 | 49 | 62 | -13 |
| 16 | Crystal Palace | 13.96 | 48 | 63 | -14 |
| 17 | Ipswich | 14.83 | 46 | 64 | -18 |
| 18 | Hull | 15.75 | 44 | 66 | -22 |
| 19 | Coventry | 16.46 | 43 | 67 | -25 |
| 20 | Bournemouth | 18.50 | 38 | 71 | -33 |

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
