# Path evolution trial — Opta blend vs path-only

Diagnostic (no MILP, no likelihood gate). Compares cross-strand fan-out of match λ and a tracked club's attack rate under:

1. **opta_blend** — now the locked Opta-rank baseline (`sample_match_from_opta_ranks`; no path λ blend). Legacy name kept for the trial artifact.
2. **path_only** — exploratory only: Opta seeds `ClubPath` with prior `n0`; fixtures use path λ only; `k ∼ M`; observe. Rejected for cascade_long λ.

- decision_gw=6 · horizon GW8–19
- n_strands=40 · seed=7 · n0=4.0 · track=arsenal

## Fan-out by horizon offset

| off | GW | blend λ SD | path λ SD | blend att SD | path att SD | path n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 0.0359 | 0.0000 | 1.4242 | 0.2893 | 5.0 |
| 1 | 9 | 0.0815 | 0.1460 | 1.1080 | 0.3078 | 6.0 |
| 2 | 10 | 0.0809 | 0.1624 | 1.0098 | 0.4160 | 7.0 |
| 3 | 11 | 0.0616 | 0.2811 | 0.9044 | 0.4033 | 8.0 |
| 4 | 12 | 0.0772 | 0.2136 | 0.7530 | 0.4267 | 9.0 |
| 5 | 13 | 0.0709 | 0.3928 | 0.6676 | 0.4242 | 10.0 |
| 6 | 14 | 0.0610 | 0.3121 | 0.6081 | 0.4120 | 11.0 |
| 7 | 15 | 0.0605 | 0.2755 | 0.5706 | 0.4222 | 12.0 |
| 8 | 16 | 0.0608 | 0.3092 | 0.5198 | 0.4351 | 13.0 |
| 9 | 17 | 0.0493 | 0.3715 | 0.4951 | 0.4643 | 14.0 |
| 10 | 18 | 0.0523 | 0.4131 | 0.4581 | 0.4641 | 15.0 |
| 11 | 19 | 0.0563 | 0.4111 | 0.4524 | 0.4722 | 16.0 |

## Growth (early → late)

| metric | opta_blend | path_only |
| --- | --- | --- |
| λ SD | 0.0359 → 0.0563 (1.57×) | 0.1460 → 0.4111 (2.81×) |
| arsenal att SD | 1.4242 → 0.4524 (0.32×) | 0.2893 → 0.4722 (1.63×) |

## Read

- **path_only:** λ SD and att SD **rise** with horizon — strand outcomes compound; no `z(h)` needed.
- **opta_blend:** λ SD stays tiny; att SD **falls** — Opta re-anchor pulls strands back toward the prior and suppresses fan-out.
- `n0` only protects the Opta seed from a one-match wipe; it is not a fixture blend weight. `n0=0` wipes att to the first rolled scoreline.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.models.path_evolution_trial --n 40 --n0 4
```
