# Stage 7 — Historic xMi backtest

**Canonical xMi = 3-GW rolling mean of minutes** (history only).
LLM / team-news overlay is deferred; keep backtesting other point drivers first.

Includes 0-minute GW rows from Vaastav. Features use prior GWs only.

- Raw player-GW rows: **29338**
- Backtest rows (prior GWs ≥ 3): see tables

## Models

| name | definition |
|---|---|
| `xmi` / `xmi_roll3` | **primary** — mean minutes over last 3 GWs |
| `xmi_roll2` | comparator (lowest MAE; more twitchy) |
| `xmi_roll5` | comparator |
| `xmi_roll8` | comparator |
| `xmi_expanding` | comparator |
| `xmi_mixture` | comparator — P₃(start)·E[mins\|start] + (1−P₃)·E[mins\|bench] |
| `xmi_blend` | comparator — 0.5·roll3 + 0.5·mixture |

## Overall minutes backtest

| model | n | MAE | RMSE | R² | corr | mean y | mean pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| xmi | 26819 | 11.42 | 23.60 | 0.601 | 0.787 | 25.2 | 25.3 |
| xmi_roll2 | 26819 | 10.74 | 23.42 | 0.607 | 0.794 | 25.2 | 25.2 |
| xmi_roll5 | 26819 | 12.58 | 24.38 | 0.574 | 0.767 | 25.2 | 25.3 |
| xmi_roll8 | 26819 | 13.56 | 24.99 | 0.553 | 0.751 | 25.2 | 25.4 |
| xmi_expanding | 26819 | 14.98 | 25.64 | 0.529 | 0.733 | 25.2 | 25.8 |
| xmi_mixture | 26819 | 12.02 | 23.88 | 0.592 | 0.779 | 25.2 | 25.3 |
| xmi_blend | 26819 | 11.68 | 23.62 | 0.600 | 0.785 | 25.2 | 25.3 |

## Primary (`xmi` = roll3) by position

| pos | n | MAE | RMSE | R² | corr |
|---|---:|---:|---:|---:|---:|
| GKP | 3092 | 4.57 | 17.33 | 0.785 | 0.889 |
| DEF | 8791 | 13.52 | 26.95 | 0.551 | 0.757 |
| MID | 11983 | 11.77 | 22.98 | 0.585 | 0.777 |
| FWD | 2953 | 10.93 | 20.88 | 0.623 | 0.799 |

## Start classification (`roll3_start_rate` ≥ 0.5)

- n=26819
- accuracy=**0.873**
- Brier=**0.099**
- base start rate=0.281

## Plots

- `data/plots/xmi_backtest.png`
- `data/plots/xmi_by_position.png`

## Output

- `data/processed/xmi_historic.csv` — use column **`xmi`**

## Decision

- Locked **roll3** as xMi (better than roll5; less twitchy than roll2).
- Do **not** add LLM/news until other scoring channels are backtested historically.
