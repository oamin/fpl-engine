# FPL engine

Two live managers share the 2026/27 rules in `src/rules/fpl_2026.py`. They do not share a score.

| Manager | Code | What it reads | What it locks |
| --- | --- | --- | --- |
| numagent | `src/live/scorer.py`, `src/live/half_plan.py`, `src/eval/decision_spec.py` | T−1 minutes, Exchange lines, bootstrap prices | `data/predictions/2026-27/week_freeze.jsonl` |
| stragent | `src/str_agent/` | The entry ledger, audited notes, and the string dossier | `data/predictions/2026-27/string_agent_freeze.jsonl` |

Numagent scores a squad. `score_xp` is the shadow column. From Gameweek 6 the wired live primary in `decision_spec.py` is `ep_next`. The Gameweek 6 row in the numeric ledger is the `score_xp` wildcard that was submitted. `GW6_CHIP_COUNTS` is false, so that chip is a logged judgement.

Stragent writes prose from the dossier and returns a legal squad. It does not read `score_xp`. The official Gameweek 6 plan is `data/predictions/2026-27/string_plans/gw06.json`.

```bash
python3 -m src.live
python3 -m src.str_agent --entry 2632584 --hold
python3 -m src.str_agent.trawl --gw 6 --deadline 2026-10-10T10:00:00Z
python3 -m src.live.session --gw N --deadline <utc>
python3 -m src.str_agent.session --gw N --deadline <utc>
```

The next deadline uses those two session commands. Numagent compiles `gwNN/xmi_t1.csv` from `news_packets`, reads `gwNN/betfair_t1` for match odds, anytime prices, and outright ranks, writes `score_xp` for the deadline week and `forecast_xp` for later weeks, and chooses the squad on `score_xp`. `--lock` appends one numeric ledger row when the clock is before the deadline. That row keeps the decision, the untouched eleven, the one-transfer eleven, and both scores for those elevens.

Stragent polls the dossier feeds, writes the prompt from that dossier and earlier notebook rows, and waits. `--reply` is the agent's JSON. From Gameweek 7 the starting squad is the previous plan. `--lock` appends week 0 of that plan. Each session leaves `gwNN/session_numagent.json` or `gwNN/session_stragent.json`, with file hashes, the model iteration, `locked`, and `realised` null until the gameweek is read. The iterations live in `data/predictions/2026-27/model_registry.jsonl`. A lock stops when the working tree of the model files differs from the current row.

Collectors and storage are mapped in `data/README.md`. Locked sessions are indexed in `data/predictions/2026-27/LOCKS.md`. Older stage reports live under `reports/`. `reports/SUPERSEDED.md` names which of those totals are historical.

Rules and the collaboration protocol are in `PROJECT_OUTLINE.md`.
