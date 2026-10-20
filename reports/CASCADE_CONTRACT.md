# Cascade contract — Numagent 2.0 surfaces

**Product:** Numagent 2.0 (`score_xp_v2`). **Archive:** `ARCHIVE_NUMAGENT_1_0.md`.  
**Plan:** `CASCADE_PLAN.md`. **Scorer checklist:** `SCORE_XP_V2_CHECKLIST.md`.

Two branches, one principle: **cascade_short** (priced t:t+1) and
**cascade_long** (Opta t+2…). Neither re-enters the live Odds/FD board factory
for scenario MILP — they consume the surfaces below.

## Surfaces

| Surface | Module | Cascade calls |
| --- | --- | --- |
| **player_pool** | `src/models/player_pool.py` | FFScout ∩ minutes; shared sticky ids |
| Rates / α / shares (as-of) | `src/markets/numagent_xp.py` | freeze priors before scenario draws |
| Match → side moments | FD/Odds or Opta λ → Dixon–Coles | short: priced; long: Opta bands |
| Player xP (mean) | `numagent_xp.score_team_side` via `cascade_score_v2` | λ→xP bridge |
| Player xP (short scenario) | `short_horizon_generative` | priced t:t+1 strands only |
| Transfers (single week) | `src/strategy/layer3_week.py` | cascade_short t / t+1 |
| Horizon path MILP | `src/strategy/layer4_week.solve_horizon_transfers` | **μ injected**, not projected |
| μ injection | `src/strategy/layer4_mu_inject.py` | `pools_from_mu_boards` → L4 |
| Branch entries | `cascade_short.py` / `cascade_long.py` | thin public surfaces |

## Locked principles

1. **Scorer** — `SCORE_NAME = score_xp_v2`. No `xp_on_pot` / `compute_xp` on cascade strands.
2. **Shares** — `SEED_SHARES = False` until a gated screen promotes otherwise (parked 2026-10-20).
3. **Short horizon (t : t+1)** — priced FD (or live book) λ; frozen as-of rates.
   - **Mean layer:** `score_team_side` (moment xCS/xGC); may cross-check `project_week_score_xp_v2`.
   - **Scenario strands (default):** λ ±15% → `k ∼ M` → lineup cov → attack dice → CS/GC from rolled GA; soft priors = locked `score_xp_v2` DefCon/bonus/cards/saves helpers (`protocol.cascade_short_generative`). Firm XI deferred; bonus-from-roll parked.
   - Path FT uses injected μ + L3/L4, not a second call into the priced factory.
4. **Long horizon (t+2…)** — every fixture baselines from Opta place → GF/GA (±MAE) → λ → Dixon–Coles `M` → `k ∼ M` (1st vs 20th keep distinct rates; no ClubPath λ blend) → generative player xP equivalent to cascade_short (**props off**) via `score_pool_generative_on_scenarios` (`protocol.cascade_long_match_outcome`).
5. **Hand-off** — priced weeks may `observe` onto `ClubPath` before Opta roll (`strand_handoff`); plumbing only — ClubPath is unused for long-horizon λ.
6. **L4** — solver is column-agnostic (`mu_col`). Priced live default stays `build_mupost_priced_horizon` / live Odds boards. Cascade builds μ DataFrames and calls `pools_from_mu_boards`.

## Do not

- Import private `_attach_store_ids` / `_align_to_week0` from cascade modules — use `layer4_mu_inject`.
- Fit on the 2026/27 holdout or reopen archived `score_xp` screens as the cascade scorer.
- Add a new stage per day; one pre-registered likelihood comparison at a time.

## Parked under v2 (do not re-screen without a new plan)

Share seed, assist α_A = team xA/xG, match-band z, player finish scales, Opta rank→GF/GA map, cascade_short bonus-from-roll — see `.cursor/context.md` / `ideas.md`.
