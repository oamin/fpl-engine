Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Discrete M-event predictive likelihood vs Gaussian-on-total

Pre-registered: `protocol.m_discrete_event_likelihood_screen`.

Baseline: Gaussian(`total_points` | FD centreline `score_xp_v2` μ, σ=3).
Treatment: factored log score from the **same** locked μ — Gaussian G/A/rest,
Bernoulli CS via \(P(\mathrm{CS})\) from \(M\), categorical \(\lfloor GA/2\rfloor\) from \(M\).

GW8–38 · wall 168.1s · n=22522

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Result (treatment − baseline)

- Mean Δ log-score / GW: **-4.6171** [-4.6323, -4.6018]
- Covers 0: **False**
- Verdict: **park_clear_lose**
- Mean layer / defaults **unchanged**.

| Season | n | mean Δ log | mean log_base | mean log_fact |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | 5602 | -4.6499 | -2.6655 | -7.3161 |
| 2023-24 | 5474 | -4.5987 | -2.6940 | -7.2921 |
| 2024-25 | 5667 | -4.6190 | -2.6259 | -7.2459 |
| 2025-26 | 5779 | -4.5999 | -2.6995 | -7.3007 |

## Component mean logs (treatment, pooled)

- log_G: -2.1840
- log_A: -2.0750
- log_CS: -0.4930
- log_GC: -0.4128
- log_rest: -2.1239

## Notes

- Positive Δ = factored M-event likelihood beats Gaussian-on-total.
- Mode/median boards were not used.
- Eligible: minutes>60, ≥3 prior apps, FD-priced weeks.
- **Scale note:** summing five component log scores is not commensurate with
  one Gaussian on totals (treatment mean log ≈ −7.3 vs baseline ≈ −2.7). The
  registered pair therefore **clearly loses** for structural reasons, not
  because discrete CS/GC are worse on their own terms.

## Diagnostic (not the promote pair)

Same 5-term skeleton; only CS/GC laws change — Gaussian(`xp_cs`/`xp_gc`, σ=3)
vs Bernoulli / categorical from \(M\):

- Mean Δ log / GW (discrete − Gaussian CS/GC): **+3.2309** [+3.2119, +3.2503]
- Covers 0: **False** (lo > 0)
- Isolates the discreteness claim under a commensurate score.

| Season | mean Δ (disc − gauss CS/GC) |
| --- | ---: |
| 2022-23 | +3.2013 |
| 2023-24 | +3.2389 |
| 2024-25 | +3.2417 |
| 2025-26 | +3.2420 |

Mean layer still unchanged. Follow-up (if pursued): re-register the
same-skeleton pair as the primary gate — do not promote off this diagnostic alone.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.m_discrete_event_likelihood_screen
```
