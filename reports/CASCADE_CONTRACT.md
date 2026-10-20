# Cascade contract — Numagent 2.0 surfaces

**Product:** Numagent 2.0 (`score_xp_v2`). **Archive:** `ARCHIVE_NUMAGENT_1_0.md`.

Cascade must not re-enter the live Odds/FD board factory. It consumes the
thin surfaces below.

## Surfaces

| Surface | Module | Cascade calls |
| --- | --- | --- |
| Rates / α / shares (as-of) | `src/markets/numagent_xp.py` | freeze priors before scenario draws |
| Match → side moments | FD AH+OU or Opta λ → Dixon–Coles | short: FD ±15%; long: Opta bands |
| Player xP | `numagent_xp.score_team_side` via `src/markets/cascade_score_v2.py` | only λ→xP bridge for long horizon |
| Transfers (single week) | `src/strategy/layer3_week.py` | short-horizon t / t+1 |
| Horizon path MILP | `src/strategy/layer4_week.solve_horizon_transfers` | **μ injected**, not projected |
| μ injection | `src/strategy/layer4_mu_inject.py` | `pools_from_mu_boards` → L4 |

## Locked principles

1. **Scorer** — `SCORE_NAME = score_xp_v2`. No `xp_on_pot` / `compute_xp` on cascade strands.
2. **Shares** — `SEED_SHARES = False` until a gated screen promotes otherwise (parked 2026-10-20).
3. **Short horizon (t : t+1)** — priced FD (or live Odds) λ; frozen as-of rates; optional λ ± band scenarios. Mean path may cross-check `project_week_score_xp_v2`; **scenario / path FT uses injected μ + L3/L4**, not a second call into the priced factory.
4. **Long horizon (t+2…)** — Opta ranks → GF/GA → λ → `ClubPath.observe` → `cascade_score_v2` / `score_pool_on_lam_pots`.
5. **Hand-off** — priced weeks may `observe` onto `ClubPath` before Opta roll (`strand_handoff`); plumbing until a path-V / player-log arm is registered.
6. **L4** — solver is column-agnostic (`mu_col`). Priced live default stays `build_mupost_priced_horizon` / live Odds boards. Cascade builds μ DataFrames and calls `pools_from_mu_boards`.

## Do not

- Import private `_attach_store_ids` / `_align_to_week0` from cascade modules — use `layer4_mu_inject`.
- Fit on the 2026/27 holdout or reopen archived `score_xp` screens as the cascade scorer.
- Add a new stage per day; one pre-registered likelihood comparison at a time.

## Parked under v2 (do not re-screen without a new plan)

Share seed, assist α_A = team xA/xG, match-band z, player finish scales, Opta rank→GF/GA map — see `.cursor/context.md` / `ideas.md`.
