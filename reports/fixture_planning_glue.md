Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Fixture planning glue — captain / stack diagnostics

Pre-registered decision screen on fixture-covariance draws (`protocol.fixture_planning_glue`). Not a μ likelihood gate; not a climb.

- Min history GW: **8**
- Scenarios / fixture: **100**
- Haul threshold: **≥ 8** pts
- CVaR: mean of top **20%** draw scenarios
- Captain multiplier: **×2**
- As-of poison checks: **PASS**

## Contrasts (GW-mean Δ, cluster interval)

| contrast | mean | lo | hi | covers 0? |
| --- | ---: | ---: | ---: | --- |
| `captain_haul_minus_captain_mu` | -0.031 | -0.313 | +0.287 | yes |
| `captain_cvar_minus_captain_mu` | -0.027 | -0.356 | +0.277 | yes |
| `stack_same_minus_diversify` | +0.550 | +0.262 | +0.855 | no |

## By season

| season | as-of | fixtures | GWs | frac haul≠μ | frac CVaR≠μ | mean Δ haul−μ | mean Δ CVaR−μ | mean Δ stack−div |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | PASS | 306 | 31 | 17% | 22% | +0.010 | +0.493 | +0.658 |
| 2023-24 | PASS | 310 | 31 | 21% | 23% | +0.239 | -0.129 | +0.502 |
| 2024-25 | PASS | 310 | 31 | 17% | 19% | -0.061 | -0.158 | +0.572 |
| 2025-26 | PASS | 298 | 31 | 29% | 33% | -0.313 | -0.312 | +0.469 |

## Verdict

Planning-glue screen complete. captain_haul vs μ inconclusive (covers 0); captain_cvar vs μ inconclusive; same-fixture stack beats diversify.

No clear captain override — μ stays default; optional live shadow only. Proceed to Layer 4 horizon when ready.

JSON: `data/processed/fixture_planning_glue.json`
