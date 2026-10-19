Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Item 5 — Rank tracker (expanding GF/GA)

Pre-registered. Candidate: leakage-free expanding club GF/GA → Dixon–Coles match pots (no finishing ranks). Controls: prior-season ranks and final-table oracle via the existing OLS rank→GF/GA map.

Pass: beat prior_rank on match MSE in every closed season **and** eligible player-GW log interval (tracker − prior) strictly above zero. If that passes but <50% of the oracle−prior log gap closes, stop (ceiling thin).

## Decision

**keep_rank_map_park_tracker**

- Beat prior MSE all seasons: **False**
- Beat prior log (interval lo>0): **False** (-0.00216 [-0.00736, +0.00314])
- Fraction of oracle−prior log gap closed: **-0.136** (≥0.5? False)

## Per season

| season | MSE tracker | MSE prior | MSE oracle | log tracker | log prior | log oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 1.727 | 1.718 | 1.500 | -2.54866 | -2.55621 | -2.53541 |
| 2023-24 | 1.724 | 1.553 | 1.528 | -2.54614 | -2.53511 | -2.52961 |
| 2024-25 | 1.581 | 1.415 | 1.337 | -2.51604 | -2.51204 | -2.49872 |
| 2025-26 | 1.474 | 1.296 | 1.226 | -2.54452 | -2.54510 | -2.53427 |

## Outputs

- `data/processed/rank_tracker_screen.json`
- `data/processed/rank_tracker_logscore.csv`

Reproduce: `PYTHONPATH=. python3 -m src.eval.rank_tracker`.
