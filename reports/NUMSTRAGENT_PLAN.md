# numstragent — plan (come back here)

**Named:** 2026-11-01  
**Revised:** 2026-11-01 — Layer 1 = number cruncher only  
**Status:** parked plan — not a protocol screen; live product architecture on top of locked Numagent / joint path  
**Hist minutes note:** `l4_opta_joint_path_xmi_sticky` **parked clear lose** (−2.56/GW). Keep expanding as-of m on Layer 1; scout live needs richer “out N weeks” tags, not one-blank sticky.

**Scout-m shares locked (PI 2026-11-01):** when scout rolling-3 m is applied, **freeze** sG/sA from expanding backroom m — [`SCOUT_M_SHARE_FREEZE_LOCK.md`](SCOUT_M_SHARE_FREEZE_LOCK.md).

## Idea

Manager with a backroom, not LLM-replaces-MILP. **Joint path is only Layer 1** — μ boards + transfer MILP reference. Risk (σ), chips, and (for Layer‑1 isolation) captaincy sit above or beside it; the scout owns the minutes portfolio.

| Layer | Name | Job | Sovereign? |
| --- | --- | --- | --- |
| 1 | Backroom | L4+Opta **μ** boards + joint-path MILP **reference** (FT/bank state; no chip search; no σ in objective) | Yes for μ / transfers |
| 2 | Scout | Squad news → expected XIs / multi-GW **xmi**; props → haul shortlist | Yes for minutes portfolio |
| 3 | Coach | LLM: salient brief + GW calendar → multi-GW plan (deltas vs reference) | Proposes only |
| 4 | Accountant | Legality / hits / chips / XI repair (`fpl_2026` + path commit) | Yes for rules |
| 5 | Overseer | Human (PI) final gate when needed | Optional later |

## Layer 1 boundary (joint path)

**In:** centreline μ (`mu_post`), Opta-far decay, every-GW path transfers, FT/bank accounting.  
**Out of Layer 1 objective / screens:** chips; `σ_μ` / μ-band / risk tilts (parked); captain double — **product still has `value_captain=true` locked**, but new Layer‑1 hist climbs for minutes use **`value_captain=false`** so xmi is not confounded with cap. Captaincy returns as coach/overseer (or a separate cap screen), not as backroom μ work.

## Design locks (do not blur)

1. **Brief, don’t dump** — MILP week-0 + alts, scout xmi vs owned, prop outliers, FT/bank; σ/chips only as tagged facts for coach, not MILP inputs.
2. **Coach proposes deltas; accountant is sovereign.**
3. **Scout owns minutes; backroom owns μ** — no coach-invented λ.
4. **Science for μ/transfers stays Layer 1 climbs**; numstragent live until frozen brief+model.
5. **Not μ-band** — parked; variance ≠ news/xmi.

## Next hist work (before live scout polish)

**Goal:** feed joint path a **deadline-causal minutes portfolio** that a scout could have known — not oracle actuals.

1. Pre-register as-of **xmi proxy** on closed seasons (e.g. sticky blank / last‑k minutes / role flags → multi‑GW `m` on far boards; week‑t tighter).
2. Climb vs product expanding as-of mean minutes on locked path; **`value_captain=false` both arms**; no chips; no σ.
3. Compare to oracle@t ceiling (+2.07) as sanity only — promote only if proxy clears 0 without leakage.
4. Then wire the same schema into live scout (FFScout / news “out N weeks”).

## Build order (numstragent live)

1. Hist one-blank sticky **parked**.  
2. **Live scout v1 (started 2026-11-01):** FFScout → sparse minutes packet on expanding baseline — [`SCOUT_LIVE_FFSCOUT.md`](SCOUT_LIVE_FFSCOUT.md) · `src/live/scout_minutes.py`. Out/Banned/Doubts step-0 only; share freeze; `--scout-packet` on live pick.  
3. Structured multi-GW tags (`out_for_n_gws` / permanent rotation) + Opta-far per-step maps.  
4. Structured brief schema (Layers 1–2).  
5. Coach deltas + accountant.  
6. Overseer on live GW; chips only via coach/accountant, never Layer‑1 search.

## Out of scope

- σ / dice / risk in joint-path objective  
- Chip calendar inside Layer‑1 MILP  
- Replacing μ with generative cascade as mean  
- Odds API / frozen holdout as live book  

## Related

- Strategiser lock: `reports/JOINT_PATH_OPTA_DECAY_LOCK.md` (captain still on product; isolate off for xmi climbs)  
- Minutes ceiling: `reports/l4_opta_joint_path_deadline_causal_minutes_gate.md`  
- Parked live multi-GW xmi: `.cursor/ideas.md`  
- μ-band park: `reports/l4_opta_joint_path_mu_band_roll_gate.md`
