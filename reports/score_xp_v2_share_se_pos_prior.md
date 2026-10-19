Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# `score_xp_v2` share-SE shrink (position×minutes prior) vs raw

Pre-registered: `protocol.score_xp_v2_share_se_pos_prior`.

Treatment shrinks **sG/sA** toward ``s0 ∝ m · w(pos)`` with locked closed-season per-90 position weights (MID=1; GKP goals 0). Addresses sparse early-season CB goal monopoly without all-minutes invent.

Formula: `s0∝m·w(pos); w_G={GKP:0,DEF:0.255,MID:1,FWD:2.467}; w_A={GKP:0.042,DEF:0.396,MID:1,FWD:0.954}; τ²=median(Δs²); K=τ²/(τ²+Δs²); s̃=Ks+(1−K)s0; renormalise`

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed** (rates/priors GW `< t`; minutes oracle).

## Result (treatment − baseline)

- Mean Δ log-score / GW: **0.0195** [0.0158, 0.0234]
- Covers 0: **False**
- Verdict: **promote**
- Pooled player-fixture rows: 27752

| Season | n | mean Δ log | MAE raw | MAE shr |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | 6937 | +0.0195 | 2.295 | 2.279 |
| 2023-24 | 6992 | +0.0176 | 2.330 | 2.321 |
| 2024-25 | 6931 | +0.0203 | 2.271 | 2.264 |
| 2025-26 | 6892 | +0.0207 | 2.381 | 2.353 |

## Notes

- Pool matches `score_xp_v2_screen` (minutes>60, FD AH+OU, GW5–38, σ=3).
- Positive Δ = position×minutes share-SE beats raw on Gaussian log score.
- Weights locked from pooled 2022-23…2025-26 per-90 rates (not fit on holdout).
- **Action 2026-10-17:** promote → default `share_se_shrink=True`, `share_se_prior='position_minutes'`.
