# Stage 5 — Assemble player μ

μ = baseline_points + predicted residual from GO market features + prior shares.

## GO features used (from stage 3)

- GKP: defend_threat, ah_line_own
- DEF: defend_threat, p_over, ah_line_own
- MID: attack_strength, p_over, ah_line_own
- FWD: p_ah_cover

| pos | n | features | R²(resid) | corr(μ,pts) | corr(base,pts) | top−bot μ lift |
|---|---:|---|---:|---:|---:|---:|
| GKP | 646 | defend_threat,ah_line_own,prior_share_minutes | 0.015 | 0.096 | 0.042 | 0.66 |
| DEF | 2691 | defend_threat,p_over,ah_line_own,prior_share_minutes | 0.047 | 0.178 | 0.101 | 1.70 |
| MID | 2926 | attack_strength,p_over,ah_line_own,prior_share_minutes,prior_share_xg | 0.046 | 0.145 | 0.105 | 1.68 |
| FWD | 691 | p_ah_cover,prior_share_minutes,prior_share_xg | 0.031 | 0.155 | 0.166 | 1.49 |

Plot: `data/plots/stage_5_player_mu.png`

Output: `data/processed/player_mu.csv`

## Success read

- corr(μ, pts) ≥ corr(base, pts) and top−bot lift > 0 → market terms help.
- If R²(resid) ≈ 0, μ collapses to baseline (still useful for ranking by level).
