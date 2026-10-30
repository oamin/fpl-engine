Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Same-skeleton discrete CS/GC vs Gaussian CS/GC (GKP/DEF)

Pre-registered: `protocol.m_skel_cs_gc_likelihood_screen`.

Both arms: five-term log score from the **same** locked FD centreline μ —
Gaussian G/A/rest identical. Only CS/GC laws change.

- Treatment `l3_skel_disc_cs_gc`: Bernoulli P(CS) from M + categorical floor(GA/2) from M.
- Baseline `l3_skel_gauss_cs_gc`: Gaussian(P_CS | xp_cs, σ=3.0) + Gaussian(P_GC | xp_gc, σ=3.0).

Pool: GKP/DEF · GW8–38 · wall 165.6s · n=10714

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Result (treatment − baseline)

- Mean Δ log-score / GW: **+2.8276** [+2.7980, +2.8579]
- Covers 0: **False**
- Verdict: **promote**
- Mean layer / defaults **unchanged**.

| Season | n | mean Δ log | mean log_disc | mean log_gauss_skel | ΔCS | ΔGC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 2639 | +2.7961 | -7.7094 | -10.5034 | +1.6267 | +1.1673 |
| 2023-24 | 2634 | +2.8203 | -7.6688 | -10.4891 | +1.6883 | +1.1320 |
| 2024-25 | 2606 | +2.8318 | -7.6272 | -10.4576 | +1.6588 | +1.1716 |
| 2025-26 | 2835 | +2.8620 | -7.6905 | -10.5511 | +1.6517 | +1.2089 |

## Component Δ (pooled, treatment − baseline)

- Δ log_CS: +1.6563
- Δ log_GC: +1.1707

## Notes

- Positive Δ = discrete CS/GC beats Gaussian CS/GC under the same skeleton.
- MID/FWD excluded by design (null-term inflation in the prior diagnostic).
- Follows parked `m_discrete_event_likelihood_screen` (incommensurate total Gaussian).

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.m_skel_cs_gc_likelihood_screen
```

## Lock

**Locked 2026-10-30** as eval default:  · .
Mean layer unchanged.
