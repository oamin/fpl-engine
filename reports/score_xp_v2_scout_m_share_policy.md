Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Scout-m share policy — freeze vs recompute (likelihood)

Pre-registered: `protocol.score_xp_v2_scout_m_share_policy`.

Same as-of scout `m` = rolling-3. Freeze = shares from expanding `m`;
recompute = shares from scout `m`. Appear/CS/GC use scout `m` both arms.

- Pool: minutes>60, FD AH+OU, GW≥5, σ=3.0.
- Rows: 27746. Wall 202s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `share_freeze` − `share_recompute`

- Mean Δ log-score/GW: **+0.006439** [+0.004846, +0.008154]
- Covers 0: **False**
- Verdict: **promote**
- Mean xp recompute/freeze: 2.939 / 2.906

| season | mean Δ log/GW |
| --- | ---: |
| 2022-23 | +0.003966 |
| 2023-24 | +0.005285 |
| 2024-25 | +0.007990 |
| 2025-26 | +0.008443 |

## Notes

- Positive = freeze shares beat recompute under scout rolling-3 m.
- Climb follows only if this likelihood does not clearly lose.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.markets.scout_m_share_policy_likelihood_gate
```
