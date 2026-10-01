# Market calibration — 1X2 & OU vs outcomes

- Fixtures: **380** (Shin de-vig from football-data closing-ish prices)

| market | outcome | n | mean p | rate | decile r | cal R² | Brier | MAE_bin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1X2 | Home win | 380 | 0.440 | 0.426 | 0.930 | 0.854 | 0.215 | 0.053 |
| 1X2 | Draw | 380 | 0.241 | 0.274 | 0.348 | -0.114 | 0.199 | 0.062 |
| 1X2 | Away win | 380 | 0.319 | 0.300 | 0.905 | 0.736 | 0.195 | 0.056 |
| OU2.5 | Over 2.5 | 380 | 0.548 | 0.550 | 0.620 | 0.344 | 0.245 | 0.064 |
| OU2.5 | Under 2.5 | 380 | 0.452 | 0.450 | 0.606 | 0.337 | 0.245 | 0.067 |

## Plots

- `data/plots/market_cal_1x2.png`
- `data/plots/market_cal_ou.png`
- `data/plots/market_cal_compare.png`

## Read

- Points near the diagonal ⇒ well-calibrated implied probs.
- High **cal R² / decile r** with low match Brier is normal for binary events.
- Compare 1X2 home vs OU over: both should track y=p if markets are sharp.
