# numstragent — plan (come back here)

**Named:** 2026-11-01  
**Status:** parked plan — not a protocol screen; live product architecture on top of locked Numagent / joint path  
**Depends:** `joint_path_opta_decay` (+ captain); scout/xmi still thin; chips deferred in hist climbs

## Idea

Manager with a backroom, not LLM-replaces-MILP. Layer 1 stays the scientific / product commit; upper layers add tagged context MILP cannot ingest (news, rotation, props narrative).

| Layer | Name | Job | Sovereign? |
| --- | --- | --- | --- |
| 1 | Backroom | L4+Opta boards + joint-path MILP **reference** plan (μ, σ, bank, FT, chips state) | Yes for μ |
| 2 | Scout | Squad news → expected XIs / multi-GW availability; props → haul shortlist | Yes for minutes narrative |
| 3 | Coach | LLM: salient brief + GW calendar → multi-GW plan (deltas vs reference) | Proposes only |
| 4 | Accountant | Legality / hits / chips / XI repair (`fpl_2026` + path commit) | Yes for rules |
| 5 | Overseer | Human (PI) final gate when needed (chips, big hits early on) | Optional later |

## Design locks (do not blur)

1. **Brief, don’t dump** — fixed card: MILP week-0 + 2–3 alts, scout conflicts vs owned, top prop/σ outliers, FT/bank/chips, next 3–5 fixtures.
2. **Coach proposes deltas; accountant is sovereign** — LLM never owns legality.
3. **Scout owns minutes story; backroom owns μ** — no coach-invented λ / centreline rewrite.
4. **Science stays Layer 1** — closed-season gates stay pre-registered climbs / likelihood; numstragent surprises are live until a frozen brief+model arm exists.
5. **Not μ-band** — parked clear lose; variance-on-mean-boards ≠ tagged news/xmi.

## Build order (when reopened)

1. Scout multi-GW xmi carry (“out N weeks”) — hist ceiling already +2.07 oracle@t; product must stay as-of / news-causal.
2. Structured brief schema (JSON) from Layers 1–2.
3. Coach prompt + delta output contract.
4. Accountant projector (reuse path / rules).
5. Live GW overseer loop; optional later: frozen coach-vs-MILP climb.

## Out of scope until asked

- Replacing `joint_path_opta_decay` as hist product commit  
- Odds API / frozen holdout slate as live book  
- Chip search as default hist climb without a new protocol lock  

## Related

- Product strategiser: `reports/JOINT_PATH_OPTA_DECAY_LOCK.md`, `reports/JOINT_PATH_CAPTAIN_LOCK.md`  
- Minutes ceiling: `reports/l4_opta_joint_path_deadline_causal_minutes_gate.md`  
- Parked live multi-GW xmi: `.cursor/ideas.md`  
- Adjacent paper track: stragent live picks (not this stack)
