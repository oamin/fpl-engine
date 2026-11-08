# Budget trial

Gated against squad vision with squad coupling: 24.7% top 10% on the
all-minutes exam. A higher number on that exam would replace the default.

The picture keeps the locked header and player rows. A goal or assist
band is one flat color: white, red, green, or blue for 0, 1, 2, or 3.
Hidden events stay gray. The conceded-rate band is hatched after the cut.
A match head predicts the fixture total, and that total is placed onto
players with minutes in order of expected count.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Gameweek-19 top 10/20/50: 22.1%, 37.6%, 75.0%
Players: 197. Locked default: 24.7%, 41.5%, 80.1%.

Cell samples, goals, gameweeks 20-38: actual 459, drawn 464.7, covered 15.1%, on a real week 14.9%.
Budget placement, goals: drawn 533.6, height 533.6, covered 25.4%, on a real week 21.8%.
Same count on random weeks covers 20.3%.

## Remainders from gameweek 1

| revealed | decoder | actual bars | drawn bars | covered | on a real week | GK | FWD |
|---|---|---:|---:|---:|---:|---:|---:|
| 80% | cell goals | 199 | 197.6 | 15.5% | 15.6% | 1.0% | 12.7% |
| 80% | budget goals | 199 | 223.4 | 27.9% | 24.8% | 1.0% | 12.7% |
| 80% | cell assists | 192 | 199.9 | 11.3% | 10.9% | 1.0% | 12.7% |
| 80% | budget assists | 192 | 222.7 | 16.7% | 14.4% | 1.0% | 12.7% |
| 20% | cell goals | 730 | 730.5 | 14.9% | 14.9% | 1.0% | 12.2% |
| 20% | budget goals | 730 | 823.9 | 25.3% | 22.4% | 1.0% | 12.2% |
| 20% | cell assists | 707 | 762.6 | 11.1% | 10.3% | 1.0% | 12.2% |
| 20% | budget assists | 707 | 845.4 | 17.9% | 14.9% | 1.0% | 12.2% |

The default holds.
Wall clock: 67 seconds on CPU.

