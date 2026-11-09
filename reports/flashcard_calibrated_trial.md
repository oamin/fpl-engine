# Calibrated placement

The reader is the locked vision model. Each future fixture is given a
number of bars equal to the sum of its bar probabilities, rounded so
the fixtures add up to the season total. The bars go to the players
in that fixture with the highest probabilities and with minutes.

A blank zero is not scored. Precision is the share of painted bars that
land on a real event. Recall is the share of real events that were painted.
The harmonic mean is zero if every cell is left blank, and it falls when
extra bars are painted on zeros.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Gameweek-19 top 10/20/50: 24.7%, 41.5%, 80.1%
Players: 197. Locked default: 24.7%, 41.5%, 80.1%.

Gameweeks 20-38, cell samples: drawn 464.9, precision 14.8%, recall 14.9%, F1 14.9%.
Ranked placement: drawn 440.0, precision 27.0%, recall 25.9%, F1 26.5%.
Same count on random weeks: precision, recall, and F1 are 19.6%.

## Remainders from gameweek 1

| revealed | decoder | actual bars | drawn bars | precision | recall | F1 | GK | FWD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 80% | cell goals | 199 | 199.0 | 16.1% | 16.1% | 16.1% | 0.1% | 11.3% |
| 80% | ranked goals | 199 | 185.0 | 29.2% | 27.1% | 28.1% | 0.1% | 11.3% |
| 80% | cell assists | 192 | 166.9 | 11.9% | 10.3% | 11.1% | 0.1% | 11.3% |
| 80% | ranked assists | 192 | 163.0 | 12.9% | 10.9% | 11.8% | 0.1% | 11.3% |
| 20% | cell goals | 730 | 718.5 | 15.0% | 14.8% | 14.9% | 0.1% | 10.9% |
| 20% | ranked goals | 730 | 676.0 | 27.4% | 25.3% | 26.3% | 0.1% | 10.9% |
| 20% | cell assists | 707 | 624.5 | 11.4% | 10.1% | 10.7% | 0.1% | 10.9% |
| 20% | ranked assists | 707 | 593.0 | 15.2% | 12.7% | 13.8% | 0.1% | 10.9% |

The capture gate stays at 24.7%.
Wall clock: 79 seconds on CPU.

