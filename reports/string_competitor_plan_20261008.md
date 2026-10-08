# Parallel string competitor — plan

Gemini ([packets + string agent](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): a separate, prompt-native manager scored on the same `week_freeze` bar. Loose shackles on reasoning; hard shackles only on FPL legality and pre-deadline freeze.

## Intent

Almost entirely string/context based. Free to evolve prompts, narrative risk posture, and how it reads audited packets. It does **not** call `score_xp`, Poisson pots, or MILP. It competes with the numeric engine on realised points only.

## Hard invariants (not loose)

1. Emit a legal 15 / XI / C / VC / bench (quotas, ≤3 per club, budget, formation).
2. `frozen_at_utc < deadline_utc` in `string_agent_freeze.jsonl`.
3. Same bar as the numeric path: GW6–25, first formal read GW26.
4. No post-deadline context in the prompt (packets pass the same leakage gate).

## Loose (encouraged to evolve)

- System / CoT prompts, few-shots, risk language
- How it weighs injury narrative vs fixture appeal
- Chip rhetoric (still must name a legal chip or none)
- Model choice behind the runner
- Repair loops when the validator rejects a draft (re-prompt with the error)

## Target layout

```
src/str_agent/
  prompt.py       # evolving prompts
  extractor.py    # roster + prices + news_packets → markdown context
  validator.py    # legality only (reuse rules from fpl_2026)
  runner.py       # LLM call → repair → commit
  freeze.py       # append string_agent_freeze.jsonl

data/predictions/2026-27/string_agent_freeze.jsonl
```

## Freeze row (sketch)

`gw`, `deadline_utc`, `frozen_at_utc`, `provenance.model_id` / `prompt_sha256` / `context_sha256`, free-text `rationale`, `decision` (chip, transfers, XV, C/VC, bench), `accounting` (bank, cost, is_legal), `realised` (filled Monday).

## Build order (after GW6 deadline is safe)

1. ~~Scaffold `src/str_agent/` with validator + freeze (no LLM required for dry tests).~~ Done 2026-10-08.
2. ~~Wire `extractor` to `news_packets.load_gameweek_packets` + entry state.~~ Done (markdown context).
3. ~~Runner with one model; illegal drafts re-prompted ≤N times then fail closed.~~ `runner.run_once` landed; inject `call_model` for a live LLM.
4. Each deadline: commit freeze beside numeric `week_freeze` row.
5. GW26: compare realised Σ vs numeric engine and vs 1FT on the shared bar.

## Explicit non-goals

- Not a replacement for `wildcard_plan` / `price_half`
- Not allowed to skip the legality validator
- Not allowed to read realised points or post-deadline articles into context
