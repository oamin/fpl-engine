# How deep are the live minutes / news tags?

Gemini ([tag depth](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): keep FPL-dated news as the live source; free-form strings do not enter MILP/`price_half` — they compile to numeric `xmi` (and optional closed enums) with as-of citations.

## What ran for GW6

`src/live/minutes_llm.py` is **not** a web trawler. One completion sees, per flagged player:

| Field | Source |
| --- | --- |
| status / chance / news | FPL bootstrap `elements` only |
| last_minutes | player gameweek log |
| output | `xmi` 0–90 or `ask` (+ free-text `note`) |

The prompt is `data/live/minutes_prompt.txt` (2 Oct snapshot, 227 rows). The scorer reads **only** the numeric `xmi` column. The LLM `note` is diagnostic, not a decision feature.

## GW1–5 tag path (separate)

`src/live/news_tags.py` uses a **closed enum**: `firm_starter | injured | transferred | benched | ask`. Notes must be dated before the deadline; bulk same-second FPL stamps are dropped. Optional `PacketDoc` packets with citation checks. Forums are parked. Tags map to fixed minutes rules, then the climb (historical) does not take them — live path only.

## Depth rating

**Shallow but auditable.** Confidence comes from FPL’s own injury/availability line and its `news_added` time, not from multi-source presser retrieval. That is enough for hard outs and chance-scaled doubts; it is weak on rotation (“rested”, “cup-tied”, “minutes managed”) that never hits the FPL news string.

## Can rich string tags drive decisions?

Not directly. `price_half` / MILP need real numbers (`xmi`, λ, pots). Strings can be:

1. **Intermediate evidence** — detailed prose + citations, stored beside the freeze.
2. **Compiled** by a fixed rule (or a second LLM step that only emits a closed enum / `xmi`) into the minutes sheet the solver already reads.

So tags can be “quite detailed” for humans and for audit, as long as the decision surface stays numeric and as-of-gated. Free-form tags inside the objective would break determinism and the multi-week bar.

## If depth is raised later

Pre-deadline document packets with `published_at < deadline`, forced `doc_ids`, and compile-to-`xmi`. Not Reddit/forums. Not post-deadline scrapes. Not strings in the knapsack.
