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
```

Collectors and storage are mapped in `data/README.md`. Locked sessions are indexed in `data/predictions/2026-27/LOCKS.md`. Older stage reports live under `reports/`. `reports/SUPERSEDED.md` names which of those totals are historical.

Rules and the collaboration protocol are in `PROJECT_OUTLINE.md`.
