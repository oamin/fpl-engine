# Locked sessions

Two ledgers. One row per gameweek. A later week appends a row. These files are not rewritten to correct a score.

Realised points stay empty until the gameweek is read.

## Numagent

Ledger: `week_freeze.jsonl`. Writer: `src/live/freeze_week.py`. Sidecar for Gameweek 6: `gw06/lock_gw6.json`.

| GW | Frozen | Deadline | Chip on the row | Captain | Vice | Realised |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 2026-10-10T09:27:29Z | 2026-10-10T10:00:00Z | wildcard | Mbeumo `2026-27:427` | Fernandes `2026-27:426` | empty |

Formation 4-4-2. Bench order: Verbruggen, Wissa, Davis, Mainoo. The row records the wildcard that was submitted. `GW6_CHIP_COUNTS` in `src/eval/decision_spec.py` is false, so the row is a logged judgement. Minutes hash `efb2c9b0cb1be3de920b75bf0f5f5f1cace273389576971cd0b9d19ef382067d`. Exchange lines hash `98e5bb4281312953f19a18383d2d6691e34279618c900c736d39c22085bb1b0a`. Score file `gw06/20261010T091116Z.csv`.

The ledger starts at Gameweek 6. The published Gameweeks 1–5 total of 280 is the historical climb in `reports/SUPERSEDED.md`. It is not a row in this file.

## Stragent

Ledger: `string_agent_freeze.jsonl`. Writer: `src/str_agent/freeze.py`. Plan: `string_plans/gw06.json`.

| GW | Frozen | Deadline | Chip | Move | Captain | Vice | Realised |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2026-10-10T09:37:55Z | 2026-10-10T10:00:00Z | none | Scarlett to Muniz | Fernandes `426` | Haaland `411` | empty |

Bench order: Forster, Shaw, Calvert-Lewin, van Ewijk. Bank left 6 tenths. Plan sha256 `cafb5269f2cf375a12e753aa85489d363a4586ba4ee14c4baaefc666d14beccc`. The dossier for that plan is `gw06/string_sources/`.

## Research plans

These files are legal dry runs. They are not ledger rows.

| Entry | Path | Decision |
| --- | --- | --- |
| 3271409 | `reports/string_horizon_3271409/gw06.json` | João Pedro to Wood, Ødegaard captain, no chip |
| 1078627 | `reports/string_horizon_1078627/gw06.json` | White and Cherki to Saka and Hall, Saka captain, no chip |

## Holdout snapshot

`data/live/HOLDOUT_FREEZE.json` was frozen on 2026-10-06. It locks the evaluation files for Gameweeks 1–5. It is not a team selection.
