# Multistage minutes overlay — Numagent 2.0 comparison

Graded FPL availability overlay on `m` (into `score_team_side`). Diagnostic
only. Rerun 2026-10-19 on `score_xp_v2`.

## Boards

| Arm | overlay | top freq | artifact |
| --- | --- | --- | --- |
| `xi_plus_captain_ft` | on | Tarkowski 0.79 · Groß 0.76 · Haaland 0.71 | `reports/multistage_xi_captain_ft_overlay_trial.md` |
| `xi_plus_captain_ft` | off | Tarkowski 0.79 · Groß 0.76 · Tzolakis 0.76 | `reports/multistage_xi_captain_ft_trial.md` |

## Read

Overlay and no-overlay frequency tables stay close under Numagent 2.0.
Keep the overlay as default plumbing for injured / doubtful selectable-pool
players; do not treat the Δfreq as a likelihood win.
