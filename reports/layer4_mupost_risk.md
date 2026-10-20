# Layer 4 μ_post + risk — GW(t:t+2) vs L3

Week **t** = Layer-3 `mu_post`. Weeks **t+1, t+2** = projected `score_xp_v2` + same shrink.
MILP maximises **`mu_dec = mu_post + λ · risk`** (risk-seeking). Risk = within-position z of posterior SE (`K·xp_se`) and fixture-covariance draw SD. Covariance **never** replaces `mu_post`. Hard stop (no beyond-t+2).

- λ = **0.35**; w_se = 0.5; w_cov = 0.5; n_scenarios = 32
- γ = 0.9; terminal = 0

## Paired GW Δpoints (L4_risk − L3)

- Mean Δ/GW: **+1.484** [-0.710, +3.855]
- Covers zero: **True**

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | +1.065 |
| 2023-24 | +2.484 |
| 2024-25 | +0.290 |
| 2025-26 | +2.097 |

## Season totals

| season | n GW | L3 | L4_risk | Δ | L3 hits | L4 hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 31 | 1899 | 1932 | +33 | 72 | 112 |
| 2023-24 | 31 | 1813 | 1890 | +77 | 80 | 84 |
| 2024-25 | 31 | 1996 | 2005 | +9 | 76 | 76 |
| 2025-26 | 31 | 1840 | 1905 | +65 | 52 | 52 |

**Pooled:** L3 7548 · L4_risk 7732 · Δ **+184**

## Vs mean-only priced L4

Mean-only priced (`reports/layer4_mupost_priced.md`) clears L3 at **+2.90/GW**. Risk arm is weaker pooled (+184 vs +359) and covers 0; same-buy vs priced ≈19%. Risk flips **2024-25** to +9 vs L3 (priced −40) but loses elsewhere and takes more hits in 2022-23.

## Verdict

Risk-seeking L4 **inconclusive** vs L3 — park as default; keep priced μ_post mean-only. Do not retune λ on these totals.

JSON: `data/processed/layer4_mupost_risk.json`
