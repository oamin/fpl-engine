# Architectural state

Updated: 2026-10-19. **Locked stack first.** Parked hypotheses: `.cursor/ideas.md`. Stage 0–48 archaeology: root `CONTEXT.md` and `reports/SUPERSEDED.md`. Canonical index of wins: `reports/LOCKED.md`.

## Plan

Pre-deadline squad advisor under 2026/27 rules. Cursor implements and screens; Gemini proposes formulae; the PI reads gated batches. A result needs the as-of audit and paired intervals on 2022-23…2025-26 (`write_gated_report`). Odds API calls need a cost estimate and PI approval (`.cursor/rules/odds-api-quota.mdc`). The live 2026/27 holdout is frozen — do not fit on it.

## Locked stack (use this)

| Layer | What | Status | Entry |
| --- | --- | --- | --- |
| **Scorer** | `score_xp_v2` (**primary**; `score_xp` superseded) | locked 2026-10-13 · primary 2026-10-19 | `src/markets/numagent_xp.py` · `reports/score_xp_v2.md` · method lock `reports/numagent_2_0_method.md` |
| **Rates** | \(M_0=270\), \(c=0.25\) | promoted 2026-10-16 | `reports/score_xp_v2_rate_m0.md` |
| **Shares** | share-SE → position×minutes prior | promoted 2026-10-17 | `reports/score_xp_v2_share_se_pos_prior.md` |
| **μ_post** | ≡ `score_xp_v2` (shrink `mode=none`) | locked 2026-10-16 | `protocol.strategy_layer2_shrinkage` |
| **Transfers** | L3 single-deadline | locked | `src/strategy/layer3_week.py` |
| **Horizon** | L4 μ_post priced, `BOOKIE_DEPTH=2`, γ=1 | **strategiser default** | `src/strategy/layer4_week.py` · `reports/layer4_mupost_gamma1.md` |
| **Live pick** | `numagent_live_pick` (props **off**) | GW7 PI lock | `src/live/numagent_live_pick.py` |

**`score_xp` is superseded** (2026-10-19). Keep it only for historical climb replay and as the named gate baseline that lost to v2. New screens and the live path use `score_xp_v2`.

### Scorer defaults (live + historical assemble)

- AH+OU (+ BTTS when priced live) → Dixon–Coles \(M\) → pots \(G=\alpha_G xG_M\), \(A=\alpha_A xG_M\)
- α from `fpl_score_rates` (2025/26 prior + live counts; refresh after each completed GW)
- Same-season-only \(rG/rA\) (no prior-season share fill); RATE_M0=270
- Minutes: packet/tags → `xmi`; predicted-XI roles (`lineup:xi` / `lineup:rotated`)
- Share-SE shrink on; prior `position_minutes`; **anytime props off** by default
- xCS / xGC additive from the same matrix; saves / DefCon / bonus / cards as in `reports/score_xp_v2.md`
- Gate vs `score_xp` (oracle minutes, GW5–38, m>60): Δ **+0.0531** [+0.0453, +0.0609] — `reports/score_xp_v2_gate.md`

### Strategiser defaults

- L1 joint MILP on absolute μ; L2 identity attach; L3 transfers+hits; **L4 priced path** week-t = L3 board, t+1 = projected `score_xp_v2` on book pots only (never an unpriced week)
- L5 beam screened, not promoted (K=8 locked as beam width only) — L4 stays default
- Captain = max μ; stack-vs-diversify diagnostic only (`reports/fixture_planning_glue.md`)

### Live path

```
python -m src.live.numagent_live_pick --gw N
```

Prefer `--from-snapshot` / existing odds JSON. Optional `--with-props` / `--fetch-odds` are quota-gated and not the locked default.

### Latest PI lock — GW7 (2026-10-17)

[`data/predictions/2026-27/gw07/numagent_lock.json`](../data/predictions/2026-27/gw07/numagent_lock.json)

- 1 FT Davis → Affengruber · C Haaland · VC Mbeumo (PI)
- Bench: Verbruggen → Murillo → Khalaili → Mainoo
- Model: `score_xp_v2` + `share_se_pos_prior` + L4 depth-2 γ=1, props off · H-obj 123.05
- Report: `reports/numagent_live_pick_gw07.md`

## Climb constants (still in force)

Hold 1.25, switch penalty 1.0, buy gate three prior appearances and xmi≥45. Empty chip map for legacy climb replay. Constants: `src/rules/fpl_2026.py`. Climb modules that still score on `score_xp` are historical only — do not extend them; new work uses `score_xp_v2` / L4.

## Parked (do not revive without a new protocol arm)

Full list: `.cursor/ideas.md` and `reports/LOCKED.md` §Parked.

Highlights: anytime props as live default; share-SE all-minutes / scorers-only; minutes decay ρ=0.7; L4 risk/chase/window/beyond-t+2 form fills; L4 FT-DP / chip half-stop; L5 as strategiser; constant-K μ_post shrink; Betfair as live pot source on US cloud (geo-blocked — diagnostic / Mac worker only).

## Rules still in force

- Gates: as-of audit + four-season paired intervals before any claim
- Holdout freeze hashes untouched
- Fewer comparisons; objective-function search parked; captain/chip sims deferred until a pre-registered likelihood arm asks
- After a discrete sub-task: write `REPORT_STATE.json`
