# Attack roll + lineup covariance — Arsenal v Everton

N=5000, seed=0.

## Competitor pairs (Arsenal, same-position, corr≤−0.4)

- **Calafiori ↔ Hincapie** (corr -1.000)
- **Eze ↔ Tzolis** (corr -1.000)
- **J.Timber ↔ White** (corr -0.973)
- **Lewis-Skelly ↔ Bruno G.** (corr -0.907)
- **Ødegaard ↔ Merino** (corr -0.807)
- **Saka ↔ Zubimendi** (corr -0.537)
- **Gyökeres ↔ Havertz** (corr -0.426)

## Effect of lineup worlds

| | lineup cov ON | OFF |
| --- | ---: | ---: |
| Gyökeres eligible | 0.256 | (mean m gate) |
| Havertz eligible | 0.744 | |
| Both eligible same path | 0.000 | — |
| Both scored same path | 0.0000 | 0.0510 |

Top scorers/path (ON): Havertz 0.327, Saka 0.244, Tzolis 0.218, Ødegaard 0.166, Rice 0.156, Barry 0.145, Gyökeres 0.123, Gabriel 0.111

Reproduce: `PYTHONPATH=. python3 -m src.markets.attack_roll_trial --n 12 --seed 0`
