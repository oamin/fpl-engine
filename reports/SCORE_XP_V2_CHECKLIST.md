# score_xp_v2 — optimisation checklist (pre-cascade)

Confirm these before opening new cascade comparisons. Product lock:
`protocol.numagent_2_0` / `reports/LOCKED.md`.

## Decomposition in progress

| Component | Status | Module |
| --- | --- | --- |
| **`match_outcome_probs`** | **defined** | `src/markets/match_outcome.py` — bookie AH/OU/(BTTS) → `M` |
| **`match_outcome_roll`** | **defined** | same — categorical draw from `M` |
| **Player attack from `M`** | **defined** | `src/markets/player_attack.py` — `xG_M`×α×shares → `E[G]`/`E[A]` |
| Player rates / shares / α | locked (below) | `numagent_xp` (shares stay M-independent) |
| Defence from `M` (CS / GC) | still in assembler | next cut |
| Assembler `score_team_side` | locked; attack delegates | `numagent_xp` |

See `reports/MATCH_OUTCOME.md` · `reports/PLAYER_ATTACK.md`. Match call sites not fully rewired yet.

## Locked (assembler / rates — still current)

| Component | Lock | Evidence |
| --- | --- | --- |
| Assembler `score_team_side` | primary 2026-10-19 | `numagent_xp.py`, gate vs `score_xp` **+0.0531** |
| RATE_M0 = 270 | 2026-10-16 | `score_xp_v2_rate_m0.md` |
| Share-SE prior = position×minutes | 2026-10-17 | `score_xp_v2_share_se_pos_prior.md` |
| Same-season shares only | 2026-10-16 | `score_xp_v2_share_history` |
| α_G / α_A pooled prior+live | 2026-10-16 | `score_xp_v2_alphas` |
| Live anytime props **off** | 2026-10-17 | `score_xp_v2_live_props` |
| Props formula = shape π, M levels | 2026-10-20 | `prop_goal_pi` / `player_attack_from_m`; assists no overlay |
| L2 μ_post ≡ score_xp_v2 | 2026-10-16 | shrink `mode=none` |
| L3 / L4 γ=1 BOOKIE_DEPTH=2 | 2026-10-15/16 | strategiser default |

## Parked (do not reopen without a new plan)

Share-SE all-minutes / scorers-only, minutes decay ρ=0.7, props-on default,
cascade share seeding, assist α_A = team xA/xG, match-band z, finish scales,
rank→GF/GA remap — see `.cursor/ideas.md` / `CASCADE_PLAN.md`.

## Still product live (not cascade)

`numagent_live_pick` + Odds pots. Cascade consumes the same scorer; it does
not replace the live entrypoint.

## Gate for “v2 ready for cascade”

- [x] Primary scorer and assembler locked
- [x] Share / rate / α defaults locked
- [x] Props default off
- [x] Cascade contract uses `score_team_side` only (no `xp_on_pot`)
- [ ] PI sign-off that no further v2 scorer screen is required before cascade path-V

When the last box is ticked, cascade work proceeds under `CASCADE_PLAN.md`
with shared `player_pool`.
