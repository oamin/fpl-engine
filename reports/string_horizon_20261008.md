# String rolling horizon

The string agent can now freeze a three-week plan before the Gameweek 6 deadline and continue from that plan afterwards. Week 0 is the decision. The next two weeks are intentions at today's prices. A later week that is illegal saves nothing. The saved carry is the squad after the imminent week only.

Gemini accepted the formula and this implementation ([string horizon](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)). The open operational case is a legal imminent week whose later intention fails the bank or the quota until the repair budget is spent: the whole plan is refused, and nothing is written. `run_once` already allows two repairs. The file command below does not call a model and does not repair; the JSON has to be legal before it is saved.

No live model was called. The official string ledger was not written. The realised-points comparison is not in this change. The numeric side of the head-to-head stays the existing `ep_next` freeze.

## What a saved plan does

`validate_horizon` requires three weeks. Week 0 must match the decision. Each later week is a legal move from the previous week's carry. A free hit lasts one week. A wildcard does not grant a free transfer, so two transfers the week after cost 4 hits. The same chip cannot be played twice in the horizon.

Gameweek 6 may start from ojaminFC. From Gameweek 7 the start is `data/predictions/2026-27/string_plans/gw06.json`. If that file is missing, preparation stops. It does not read the live entry. The follow-up context names the paper squad and the intentions, and it refuses `score_xp`, `xp_on_pot`, and `lam_scored`.

## Deadline commands

Save a drafted JSON without touching the official ledger:

```
python3 -m src.str_agent --entry 2632584 --save-plan draft.json
```

The same command with `--commit-saved` also appends the imminent week to `data/predictions/2026-27/string_agent_freeze.jsonl`, with `provenance.plan_sha256` equal to the plan file. `frozen_at` has to be strictly before the deadline. A second row for the same gameweek is refused. `--commit` on its own still exits 2.

`--save-plan` does not call a model. The draft is a JSON object with `rationale`, `decision`, and `horizon` (three weeks, this week first). The plan and its context land in `data/predictions/2026-27/string_plans/`.

The numeric lock is unchanged: on the Mac, inside the T−1h window, capture `slot_t1.json`, run the Betfair pull, then `wildcard_plan` and the hand-built `week_freeze` row. This cloud VM still gets Betfair HTTP 403.

## Tests

`python3 -m unittest tests.test_str_agent_horizon tests.test_str_agent tests.test_str_agent_carry tests.test_str_agent_trial` — 44 tests, all passed. One fixture was corrected: selling the second goalkeeper is a quota failure, so the wildcard successor keeps both goalkeepers. The hit check is unchanged (0, then 4).
