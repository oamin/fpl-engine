# Locked sessions

Two ledgers. One row per gameweek. A later week appends a row. These files are not rewritten to correct a score.

Realised points stay empty until the gameweek is read.

## Numagent

Ledger: `week_freeze.jsonl`. Writer: `src/live/freeze_week.py`. Sidecar for Gameweek 6: `gw06/lock_gw6.json`.

| GW | Frozen | Deadline | Chip on the row | Captain | Vice | Realised |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 2026-10-10T09:27:29Z | 2026-10-10T10:00:00Z | wildcard | Mbeumo `2026-27:427` | Fernandes `2026-27:426` | empty |

Formation 4-4-2. Bench order: Verbruggen, Wissa, Davis, Mainoo. The row records the wildcard that was submitted. `GW6_CHIP_COUNTS` in `src/eval/decision_spec.py` is false, so the row is a logged judgement. Minutes hash `efb2c9b0cb1be3de920b75bf0f5f5f1cace273389576971cd0b9d19ef382067d`. Exchange lines hash `98e5bb4281312953f19a18383d2d6691e34279618c900c736d39c22085bb1b0a`. Score file `gw06/20261010T091116Z.csv`.

The morning run is the locked session file `gw06/session_numagent.json`. Its hashes are the freeze hashes. `20261010T091116Z.csv` keeps the writer columns from that morning. `load_step_scores` reads Gameweek 6 in that file as `score_xp` and Gameweeks 7 and 8 as `forecast_xp`.

The ledger starts at Gameweek 6. The published Gameweeks 1–5 total of 280 is the historical climb in `reports/SUPERSEDED.md`. It is not a row in this file.

## Stragent

Ledger: `string_agent_freeze.jsonl`. Writer: `src/str_agent/freeze.py`. Plan: `string_plans/gw06.json`.

| GW | Frozen | Deadline | Chip | Move | Captain | Vice | Realised |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2026-10-10T09:37:55Z | 2026-10-10T10:00:00Z | none | Scarlett to Muniz | Fernandes `426` | Haaland `411` | empty |

Bench order: Forster, Shaw, Calvert-Lewin, van Ewijk. Bank left 6 tenths. Plan sha256 `cafb5269f2cf375a12e753aa85489d363a4586ba4ee14c4baaefc666d14beccc`. The dossier for that plan is `gw06/string_sources/`. The morning run is the locked session file `gw06/session_stragent.json`. Its plan and context hashes are the ledger hashes. The notebook row for Gameweek 6 is in `string_plans/notebook.jsonl`.

## Research plans

These files are legal dry runs. They are not ledger rows.

| Entry | Path | Decision |
| --- | --- | --- |
| 3271409 | `reports/string_horizon_3271409/gw06.json` | João Pedro to Wood, Ødegaard captain, no chip |
| 1078627 | `reports/string_horizon_1078627/gw06.json` | White and Cherki to Saka and Hall, Saka captain, no chip |

## Next deadline

Gameweek 6 is the only row in each ledger. The next row is appended by the session commands, one gameweek at a time, while the clock is before that deadline.

```bash
python3 -m src.live.session --gw N --deadline <utc>
python3 -m src.live.session --gw N --deadline <utc> --lock --frozen-at <utc>
python3 -m src.str_agent.session --gw N --deadline <utc>
python3 -m src.str_agent.session --gw N --deadline <utc> --reply reply.json --lock --frozen-at <utc>
```

Numagent stores the minutes sheet, the `betfair_t1` book, the score file, and `gwNN/session_numagent.json`. The ledger row carries the decision squad, the untouched eleven, the one-transfer eleven, `score_xp`, and `ep_next`. Stragent stores the dossier, the prompt, the three-week plan, the notebook row, and `gwNN/session_stragent.json`. The ledger row is week 0. `realised` stays null on both until that gameweek is read. A second row for the same gameweek is refused, and a locked manifest is left in place.

## Holdout snapshot

`data/live/HOLDOUT_FREEZE.json` was frozen on 2026-10-06. It locks the evaluation files for Gameweeks 1–5. It is not a team selection.
