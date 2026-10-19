Gates: paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

As-of audit soft-pass for this closed-season screen (live lock drift): refusing report: as-of audit has not passed (the live primary is not ep_next; score_xp is not the live shadow; the published historical score is not score_xp; the live primary declared a winner; the live primary wiring flag does not match the lock; a live interval that covers zero is not undetermined; the live test does not continue through gameweek 38; the live primary does not start at gameweek 6; promotion does not wait for 20 live weeks; the decision capture is not the same-stamp t1 pair; an earlier capture is a fallback; the gameweek 6 chip is not the logged judgement; the gameweek 6 chip counts toward a chip rule; no pre-deadline capture for gameweeks 7)

# Item 5 — Rank tracker (expanding GF/GA)

Pre-registered. Candidate: leakage-free expanding club GF/GA → Dixon–Coles match pots (no finishing ranks). Player xP = `score_xp_v2`. Controls: prior-season ranks and final-table oracle via the existing OLS rank→GF/GA map.

Pass: beat prior_rank on match MSE in every closed season **and** eligible player-GW log interval (tracker − prior) strictly above zero. If that passes but <50% of the oracle−prior log gap closes, stop (ceiling thin).

## Decision

**keep_rank_map_park_tracker**

- Beat prior MSE all seasons: **False**
- Beat prior log (interval lo>0): **False** (-0.00618 [-0.01318, +0.00117])
- Fraction of oracle−prior log gap closed: **-0.325** (≥0.5? False)

## Per season

| season | MSE tracker | MSE prior | MSE oracle | log tracker | log prior | log oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1.727 | 1.718 | 1.500 | -2.53798 | -2.54060 | -2.51272 |
| 2023-24 | 1.724 | 1.553 | 1.528 | -2.57773 | -2.56891 | -2.56275 |
| 2024-25 | 1.581 | 1.415 | 1.337 | -2.54154 | -2.53164 | -2.52163 |
| 2025-26 | 1.474 | 1.296 | 1.226 | -2.56979 | -2.56720 | -2.55376 |

## Outputs

- `data/processed/rank_tracker_screen.json`
- `data/processed/rank_tracker_logscore.csv`

Reproduce: `PYTHONPATH=. python3 -m src.eval.rank_tracker`.
