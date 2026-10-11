# Locked sessions

Two ledgers. One row per gameweek. A later week appends a row. These files are not rewritten to correct a score.

Realised points stay empty until the gameweek is read.

## Numagent

Ledger: `week_freeze.jsonl`. Writer: `src/live/freeze_week.py`. Sidecar for Gameweek 6: `gw06/lock_gw6.json`.

| GW | Frozen | Deadline | Chip on the row | Captain | Vice | Realised |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 2026-10-10T09:27:29Z | 2026-10-10T10:00:00Z | wildcard | Mbeumo `2026-27:427` | Fernandes `2026-27:426` | empty |

Formation 4-4-2. Bench order: Verbruggen, Wissa, Davis, Mainoo. The row records the wildcard that was submitted. `GW6_CHIP_COUNTS` in `src/eval/decision_spec.py` is false, so the row is a logged judgement. Minutes hash `efb2c9b0cb1be3de920b75bf0f5f5f1cace273389576971cd0b9d19ef382067d`. Exchange lines hash `98e5bb4281312953f19a18383d2d6691e34279618c900c736d39c22085bb1b0a`. Score file `gw06/20261010T091116Z.csv`.

The morning run is the locked session file `gw06/session_numagent.json`. Its hashes are the freeze hashes. `20261010T091116Z.csv` keeps the writer columns from that morning. `load_step_scores` reads Gameweek 6 in that file as `score_xp` and Gameweeks 7 and 8 as `forecast_xp`. The iteration is `numagent-gw6-lock`. What that run read, and the formula that turned it into the wildcard, is `reports/gw6_numagent_audit.md`.

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

## Model iterations

`model_registry.jsonl` is append-only. The current iteration of a manager is the last row for that manager. A lock stores that iteration on the session manifest: `iteration_id`, `iteration_commit`, `locked_commit`, and `source_sha256`. A later numeric row also stores those four fields on the freeze provenance. Gameweek 6 keeps its iteration on `session_numagent.json`, `session_stragent.json`, and `lock_gw6.json`.

| Lock | Iteration | Commit | Source |
| --- | --- | --- | --- |
| Gameweek 6 numagent | `numagent-gw6-lock` | `6638dce8baee02071c3f1461f065face9bd50863` | `69da7110464578f4f546167d559bd3405bc4e225e32f1fd0bdca03a246d83c55` |
| Current numagent | `numagent-anytime-full` | `57ea53daf4ca8328f8cae524c87135da11f87ebf` | `f3c1ebce7fa42a335bde4b9f68fc2a2b3e2e96020d0c936ea5ff326dd13e2e0b` |
| Gameweek 6 stragent, and the current string code | `stragent-gw6` | `756491ed3d458fb19e595d46ae62cf5468836a6a` | `ae4de9a0ce5a0f784c353e0a8d809f52ed49c2f7fa0eb59cbe6e58f3d9fab10e` |

`numagent-gw6-lock` still passes anytime rates through the residual budget. That is the code that scored the Saturday lock. `numagent-anytime-full` applies each minutes-scaled rate in full and scales the club down only when the sum exceeds that club's lambda. `stragent-gw6` is the Saturday prompt, horizon, and carry, called as `gemini-3.8-flash-high`.

The numagent hash covers the rules, the points formula, the chip margins, the squad search, and the deadline scorer. The stragent hash covers the rules, the prompt, and the legality of a plan. Session modules, manifests, and docs sit outside the hash.

`--lock` hashes the working tree. When that hash differs from the current row, the lock stops and the ledger gains no bytes. Append a new registry row for the changed files before locking again.

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
