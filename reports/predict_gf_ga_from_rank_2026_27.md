# 2026/27 GF/GA from Opta predicted rankings only

## Does Opta project GF/GA?

**No** — not in the public Supercomputer / League Prediction output.

What Opta publishes for the **PREDICTED** table:

- predicted finishing position
- predicted / expected points
- place probabilities (title, top four/six, relegation)

What they do **not** publish with that table: season **GF**, **GA**, or **GD**.

Opta’s own description: match outcomes are modelled as W/D/L from betting odds + Power Rankings, then the remaining season is simulated for finishing places ([Opta Analyst](https://theanalyst.com/articles/premier-league-predictions-2026-27-opta-supercomputer)). That product is points/place, not a goals ledger.

Siblings (not season GF/GA forecasts):

- **ACTUAL** table — goals already scored/conceded
- **EXPECTED** (xPts) table — to-date xG/xGA → expected points from matches played

So any club-level season GF/GA here is **ours**, mapped from Opta’s predicted rank via closed-season history (2022/23–2025/26 only).

## Mapping (closed seasons → Opta rank)

- `GF ≈ 79.9 -2.26·rank` (R²=0.67, MAE=7.29)
- `GA ≈ 36.3 +1.89·rank` (R²=0.6153, MAE=6.77)
- LOSO MAE ≈ GF 8.19 / GA 7.22

## Opta table (2026-10-12) → implied GF / GA

| Rk | Club | Opta pts | Pred GF | Pred GA | Pred GD |
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

## Artefacts

- `data/processed/predict_gf_ga_from_rank_2026_27.csv`
- `reports/figures/predict_gf_ga_from_rank_2026_27.json`

