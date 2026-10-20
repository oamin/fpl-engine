# Fixture covariance — walk-forward as-of audit

Diagnostic (not a climb / likelihood gate). Competition pairs from **GW < t** only.

- Min history GW before discovery: **8**
- Pair corr threshold: **≤ -0.7**
- MC subsample: **24** fixtures × **120** scenarios / season

- As-of poison checks: **PASS**

## By season

| season | as-of | OOS pairs | frac OOS min corr<0 | median OOS | MC pairs ok | med draw min | med atk–GK | med mids | resid atk–atk | resid atk–GK |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | PASS | 83 | 80% | -0.47 | 100% | -1.00 | -0.41 | +0.30 | +0.10 | -0.32 |
| 2023-24 | PASS | 87 | 78% | -0.50 | 100% | -1.00 | -0.49 | +0.32 | +0.08 | -0.26 |
| 2024-25 | PASS | 82 | 83% | -0.52 | 100% | -1.00 | -0.52 | +0.23 | +0.06 | -0.27 |
| 2025-26 | PASS | 90 | 82% | -0.52 | 100% | -1.00 | -0.43 | +0.28 | +0.03 | -0.26 |

## Reading

- **OOS minutes:** as-of pairs should stay negatively correlated on later fixtures.
- **MC pairs ok:** fraction of as-of competition pairs with draw minutes corr < −0.2.
- **Draw atk–GK / mids:** sign checks for scoreline coupling (compare to resid columns).

## Verdict

As-of checks PASS on 4 seasons. OOS minutes: median frac neg 81%, median corr -0.51. MC competition pairs with draw min corr < −0.2: 100% (mean across seasons). MC attack–opp GK median corr -0.46 (want negative). Mechanism audit: **PASS** — proceed toward planning glue.

JSON: `data/processed/fixture_covariance_audit.json`

