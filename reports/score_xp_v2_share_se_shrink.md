Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# `score_xp_v2` share-SE shrink vs raw `score_xp_v2`

Pre-registered: `protocol.score_xp_v2_share_se_shrink`.

Treatment shrinks compositional **sG/sA** toward a **minutes prior** using share SE (`Δs`) before forming E[G]/E[A]. Not constant-K on total xp.

Formula: `s0=m/Σm on m>0; τ²=median(Δs²) on playing Δs>0 (floor 1e-4); K=τ²/(τ²+Δs²); s̃=Ks+(1−K)s0; renormalise`

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed** (rates/priors GW `< t`; minutes oracle).

## Result (treatment − baseline)

- Mean Δ log-score / GW: **0.0034** [0.0013, 0.0057]
- Covers 0: **False**
- Verdict: **promote**
- Pooled player-fixture rows: 27752

| Season | n | mean Δ log | MAE raw | MAE shr |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | 6937 | +0.0004 | 2.291 | 2.303 |
| 2023-24 | 6992 | +0.0038 | 2.332 | 2.341 |
| 2024-25 | 6931 | +0.0082 | 2.265 | 2.268 |
| 2025-26 | 6892 | +0.0011 | 2.374 | 2.376 |

## Notes

- Pool matches `score_xp_v2_screen` (minutes>60, FD AH+OU, GW5–38, σ=3).
- Positive Δ = share-SE shrink beats raw `score_xp_v2` on Gaussian log score.
- Promoted 2026-10-16: default `share_se_shrink=True` in `score_team_side`.
