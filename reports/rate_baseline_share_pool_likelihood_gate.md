Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Rate baseline + full-club share likelihood gates

Pre-registered:

- `protocol.score_xp_v2_rate_baseline_likelihood`
- `protocol.score_xp_v2_full_share_likelihood`

Published FD AH+OU `score_xp_v2` stack (minutes oracle, GW5–38, σ=3). Not a climb.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## 1. Rate baseline − published same-season rates

- Mean Δ log / GW: **+0.0024** [+0.0015, +0.0035]
- Covers 0: **False**
- Verdict: **promote**
- Rows: 27746

## 2. Full-club shares − thin (>60′) shares

- Mean Δ log / GW: **+0.0101** [+0.0087, +0.0116]
- Covers 0: **False**
- Verdict: **promote**

| Season | n | Δ rate | Δ share | MAE base | MAE rate | MAE thin | MAE full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 6937 | +0.0012 | +0.0069 | 2.279 | 2.274 | 2.326 | 2.279 |
| 2023-24 | 6992 | +0.0029 | +0.0099 | 2.321 | 2.313 | 2.369 | 2.321 |
| 2024-25 | 6931 | +0.0012 | +0.0108 | 2.264 | 2.263 | 2.322 | 2.264 |
| 2025-26 | 6886 | +0.0044 | +0.0128 | 2.350 | 2.340 | 2.408 | 2.350 |

## Notes

- Positive rate Δ = prior+season baseline beats published same-season `rG`/`rA` on Gaussian log score.
- Positive share Δ = building shares on full club `m>0` and projecting onto the >60′ pool beats shares built only among >60′.
- Promote either arm into published / centreline callers only after this gate; then run a joint-path climb sanity check.
- Wall 346s.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.markets.rate_baseline_share_pool_likelihood_gate
```
