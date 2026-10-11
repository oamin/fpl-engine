# Data

Collectors write files. The models read the paths below. A frozen holdout file is not a fallback for a missing T−1 sheet.

## Collectors

| Collector | Module | Writes |
| --- | --- | --- |
| FPL slate | `src.live.fpl_snapshot` | A local bootstrap and fixture file. The T−1 copies used by the models sit under `predictions/2026-27/gwNN/`. |
| Minutes notes | `src.live.news_packets` | `predictions/2026-27/gwNN/news_packets/` |
| Minutes sheet | `src.live.xmi` | `predictions/2026-27/gwNN/xmi_t1.csv` |
| Exchange lines | `src.live.betfair_pull`, `src.live.betfair_lines` | `predictions/2026-27/gwNN/betfair_t1/` |
| String dossier | `src.str_agent.trawl` | `predictions/2026-27/gwNN/string_sources/` |
| Entry ledger | `src.live.entry` | `entry/{id}.json` |
| History and odds | `src.ingest` | `cache/` and `processed/` |

`python3 -m src.live` collects the slate and stops until a minutes file is supplied. `python3 -m src.str_agent.trawl` stores feed titles and summaries. It does not fetch article pages, and it does not write `news_packets/`.

## Storage

| Directory | Holds |
| --- | --- |
| `predictions/2026-27/` | Pre-deadline scores, T−1 inputs, both lock ledgers, and the official string plan. |
| `predictions/2026-27/gw06/` | Gameweek 6 minutes, Exchange book, news packets, string dossier, and score files. |
| `predictions/2026-27/gw06/fpl_t1/` | Bootstrap used to compile `xmi_t1.csv`. |
| `predictions/2026-27/gw06/betfair_t1/` | T−1 match odds, anytime prices, outrights, and fixtures. |
| `predictions/2026-27/gw06/betfair_20261008/` | The 8 October Exchange pull. The live score does not fall back to it. |
| `predictions/2026-27/gw06/string_sources/` | The dossier stragent read for Gameweek 6. |
| `predictions/2026-27/string_plans/` | The official three-week string plan and the manager notebook. |
| `entry/` | Stored manager ledgers. |
| `holdout/2026-27/player_gw/` | Finished Gameweeks 1–5, one file each. |
| `live/` | Holdout snapshots. Tracked files are `gw_lines.csv`, `news_docs_gw15.json`, `HOLDOUT_FREEZE.json`, and `chip_decisions.jsonl`. The rest of this directory stays local. |
| `cache/` | Rebuildable pulls. Full-season dumps are gitignored. |
| `processed/` | Derived frames. Full-season dumps are gitignored. |
| `scratch/betfair/` | Raw Exchange books. They are not republished. |

Gameweek 6 fixtures are `gw06/betfair_t1/fixtures.json`. The numeric score reads `xmi_t1.csv` and `betfair_t1/gw_lines.csv` (`src/live/t1_inputs.py`). It refuses `data/live/xmi_gw6.csv` and `data/live/gw_lines.csv`.
