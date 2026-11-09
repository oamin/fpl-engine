# Paired sequence trial

Each hidden week is read with the other squad's match week and the two
gameweeks before it. Minutes stay visible. Goals and assists after the
cut stay masked on both cards.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Cards: train 60, test 20.
Gameweek-19 top 10/20/50 on all appearances: 18.1%, 35.1%, 71.1%
Players in that ranking: 197.

## Remainders from gameweek 1

Both clubs are scored. A bar is an appearance of at least one event.
Drawn bars are the mean of 20 samples.

| revealed | channel | actual bars | drawn bars | covered | on a real week | GK | FWD |
|---|---|---:|---:|---:|---:|---:|---:|
| 80% | goals | 199 | 238.8 | 11.3% | 9.4% | 1.9% | 12.9% |
| 80% | assists | 192 | 223.4 | 8.3% | 7.1% | 1.9% | 12.9% |
| 20% | goals | 730 | 886.4 | 11.2% | 9.2% | 1.8% | 12.7% |
| 20% | assists | 707 | 823.9 | 7.9% | 6.8% | 1.8% | 12.7% |

GK and FWD are the probability of a goal bar on a hidden appearance. The same columns on an assist row repeat that goal rate.

The 16:08 sequence reader, on this same remainder, drew 235.6 goal bars at 80% revealed with 14.5% on a real week, and 734.4 at 20% revealed with 14.9% on a real week. This paired run draws more bars and lands a smaller share of them on a real week. Its gameweek-19 top 10% is 18.1%, against 20.3% for that sequence reader.

Wall clock: 55 seconds on CPU.

