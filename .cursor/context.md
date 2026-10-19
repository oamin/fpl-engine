# Architectural state

Updated: 2026-10-19. **Numagent 2.0** is the product on main (`protocol.numagent_2_0`) — **supersedes Numagent 1.0**. Parked hypotheses: `.cursor/ideas.md`. Stage archaeology: root `CONTEXT.md` and `reports/SUPERSEDED.md`. Canonical index: `reports/LOCKED.md`.

## Plan

Pre-deadline squad advisor under 2026/27 rules. Cursor implements and screens; Gemini proposes formulae; the PI reads gated batches. A result needs the as-of audit and paired intervals on 2022-23…2025-26 (`write_gated_report`). **Numagent 2.0 pots** use The Odds API (AH+OU+BTTS; quota-aware — `.cursor/rules/odds-api-quota.mdc`). Classic climb `gw_lines` may use Betfair. The live 2026/27 holdout is frozen — do not fit on it.

## Locked stack — Numagent 2.0

| Layer | What | Status | Entry |
| --- | --- | --- | --- |
| **Scorer** | `score_xp_v2` (**primary**; `score_xp` superseded) | locked 2026-10-13 · primary 2026-10-19 | `src/markets/numagent_xp.py` · `reports/score_xp_v2.md` · method lock `reports/numagent_2_0_method.md` |
| **Rates** | \(M_0=270\), \(c=0.25\) | promoted 2026-10-16 | `reports/score_xp_v2_rate_m0.md` |
| **Shares** | share-SE → position×minutes prior | promoted 2026-10-17 | `reports/score_xp_v2_share_se_pos_prior.md` |
| **μ_post** | ≡ `score_xp_v2` (shrink `mode=none`) | locked 2026-10-16 | `protocol.strategy_layer2_shrinkage` |
| **Transfers** | L3 single-deadline | locked | `src/strategy/layer3_week.py` |
| **Horizon** | L4 μ_post priced, `BOOKIE_DEPTH=2`, γ=1 | **strategiser default** | `src/strategy/layer4_week.py` · `reports/layer4_mupost_gamma1.md` |
| **Live pick** | `numagent_live_pick` (props **off**) | GW7 PI lock | `src/live/numagent_live_pick.py` |

**Numagent 1.0 and `score_xp` are superseded** (2026-10-19). Keep 1.0/`score_xp` only for historical climb replay and as the named gate baseline. New screens and the live path are Numagent 2.0 (`score_xp_v2` + L4).

### Cascade / short-horizon alignment (2026-10-19)

Today’s pre-merge cloud cascade work (multistage captain/FT/minutes,
match-band, player finish, rank tracker, first short-horizon draft) scored
with **`xp_on_pot` / `compute_xp`** — not Numagent 2.0. Audit:
`reports/cascade_pre_numagent2_audit.md`. Those arms stay parked until
rewired onto `numagent_xp`.

**Aligned:** `src/models/short_horizon_cascade.py` — FD AH+OU → Dixon–Coles
`M` → `score_team_side` (`score_xp_v2`); λ ±15% scenarios; path L1→L3.
Mean layer corr vs L4 projector = 1.0. Screen decision
`keep_scenario_advisor` (`reports/short_horizon_scenario.md`).

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
python -m src.live.numagent_live_pick --gw N --fetch-odds   # Odds API, region=us
```

Prefer `--from-snapshot` / `data/scratch/gwNN_odds_*.json` when the book is already pulled. `--fetch-odds` refreshes AH+OU+BTTS for Dixon–Coles pots (props off unless `--with-props`). Classic climb `gw_lines` readers may still use Betfair folders; frozen `data/live/gw_lines.csv` is not a live book.

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

Highlights: anytime props as live default; share-SE all-minutes / scorers-only; minutes decay ρ=0.7; L4 risk/chase/window/beyond-t+2 form fills; L4 FT-DP / chip half-stop; L5 as strategiser; constant-K μ_post shrink; Betfair pot source on US cloud (geo-blocked — Mac worker only).

## Rules still in force

- Gates: as-of audit + four-season paired intervals before any claim
- Holdout freeze hashes untouched
- Odds API: estimate cost first; region=`us`; prefer snapshots (quota rule)
- Fewer comparisons; objective-function search parked; captain/chip sims deferred until a pre-registered likelihood arm asks
- After a discrete sub-task: write `REPORT_STATE.json`
