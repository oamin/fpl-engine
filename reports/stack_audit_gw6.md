# Gameweek 6 stack audit (sensitivity)

Gemini ([stack audit](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): pre-deadline ablation of Exchange odds, coarse outright forecast, and contextual tags. This is input sensitivity, not realised performance.

Capture: `/workspace/data/predictions/2026-27/gw06/official_20261007T220914Z.csv`.
Betfair folder ready: **False** (`/workspace/data/predictions/2026-27/gw06/betfair_20261008`).

## Arms

| Arm | Config | Owned GW6 Σ | Rebuild−held GW6 | Chip | Line weeks |
| --- | --- | ---: | ---: | --- | --- |
| arm0_baseline | Odds API + rolling minutes + GW7 copy | 52.04 | 12.60 | wildcard | [6, 7] |
| arm1_tags | Odds API + contextual tags + GW7 copy | 54.86 | 12.56 | wildcard | [6, 7] |
| arm2_betfair | Betfair lines + tags + GW7 copy | — | — | skipped | Betfair derived folder missing: /workspace/data/predictions/2026-27/gw06/betfair_20261008 |
| arm3_full | Betfair lines + tags + outright forecast | — | — | skipped | Betfair derived folder missing: /workspace/data/predictions/2026-27/gw06/betfair_20261008 |

## Deltas vs arm0 (not performance)

### arm1_tags

Owned GW6 Σ Δ = +2.822. Rebuild gap Δ = -0.036666666666668846. Chip = wildcard.
Copy note: GW8 repeats GW7 and has no 1X2 of its own.

| Player | Baseline | Arm | Δ |
| --- | ---: | ---: | ---: |
| van Ewijk | 0.234 | 3.016 | +2.782 |
| Calvert-Lewin | 2.753 | 2.770 | +0.017 |
| Cherki | 3.367 | 3.382 | +0.015 |
| Ødegaard | 5.025 | 5.012 | -0.013 |
| Rogers | 5.736 | 5.744 | +0.008 |
| Barnes | 3.815 | 3.823 | +0.008 |
| Shaw | 3.607 | 3.612 | +0.005 |

## Sync needed for Exchange arms

On the Mac, commit the derived files under `data/predictions/2026-27/gw06/betfair_20261008/` (`gw_lines.csv`, `outrights_ranks.json`, `betfair_to_score.json`, `betfair_meta.json`). Do not commit `.env` or `data/scratch/betfair/`. Then re-run:

```bash
python3 -m src.live.stack_audit
```

## What this is not

- Not a claim that higher xP is better.
- Not a chip-rule test (the 114 wildcard lead is separate).
- Not a historical Exchange backtest (no books on file).
- Realised XI / Spearman wait until GW6 points are final.
