# Player outcome spread (σ_out)

`xp_se` = estimation SE of μ (shrink only). `sigma_out` = predictive SD of points around μ (scenario idio only). Do not mix.

- Rows: **27562**; with ≥3 prior eligible residuals: **22358**
- Blank / haul: **≤2** / **≥10** pts (played ≥60′)
- Player SD full weight at n≥**6**
- μ: **score_xp_v2 (non-shrunk)**

| position | n | mean σ_out | median σ_out | mean P90 | blank rate | haul rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DEF | 8611 | 2.91 | 2.78 | 6.43 | 0.625 | 0.037 |
| FWD | 2113 | 3.58 | 3.52 | 8.42 | 0.573 | 0.090 |
| GKP | 2252 | 2.66 | 2.69 | 6.63 | 0.545 | 0.044 |
| MID | 9382 | 2.94 | 2.89 | 6.95 | 0.530 | 0.079 |

## MID μ∈[3.5, 5.5]: low vs high σ_out (sanity)

- Low-σ mean σ_out **2.16** (realised resid SD **3.26**, n=814)
- High-σ mean σ_out **4.98** (realised resid SD **3.42**, n=814)

High-σ bucket should show larger realised residual SD if σ_out ranks volatility. Here Δ realised SD = +0.16 (modest in this μ band — σ_out is still the right draws object when player histories differ sharply).

## Anti-conflation

- `xp_se` → EB shrink only (unchanged).
- `sigma_out` → scenario idio only (`draw_fixture`).
- Do not subtract either from μ; do not fold σ_out into `xp_se`.

Not a forecast gate.

CSV: `data/processed/outcome_spread_closed.csv`

