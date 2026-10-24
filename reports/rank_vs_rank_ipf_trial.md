# Rank×rank IPF λ trial (LOSO)

As of `2026-10-24T20:56:43Z` (UTC).

## Setup

- Place map: **rankmean** GF/GA, LOSO (exclude scored season).
- Oracle final ranks on each closed E0 season.
- **Product** (locked cascade_long): `λ_h = GF_h·GA_a / (38·L)`.
- **IPF**: same product prior, RAS until round-robin GF/GA margins match place totals (tol 1e-6).
- Diagnostic only — does **not** change `cascade_long`.

## Overall (4 seasons, n=1520)

| map | MAE tot | bias tot | MAE λ_h | bias λ_h | corr(λ_h, hg) |
|---|---:|---:|---:|---:|---:|
| product | 1.344 | +0.007 | 0.969 | -0.138 | 0.340 |
| IPF | 1.345 | +0.000 | 0.969 | -0.141 | 0.339 |

## Per season

| season | product MAE | IPF MAE | ΔMAE | product margin err | IPF margin err | iters |
|---|---:|---:|---:|---:|---:|---:|
| 2022/23 | 1.415 | 1.416 | +0.001 | 2.19 | 4.01e-07 | 3 |
| 2023/24 | 1.384 | 1.387 | +0.003 | 1.80 | 4.20e-07 | 3 |
| 2024/25 | 1.287 | 1.288 | +0.001 | 2.03 | 4.80e-07 | 3 |
| 2025/26 | 1.291 | 1.290 | -0.001 | 2.02 | 5.96e-07 | 3 |

## Example λ (LOSO profile excl. 2025/26)

| ranks | product (λ_h, λ_a, tot) | IPF (λ_h, λ_a, tot) |
|---|---|---|
| 1v20 | (3.70, 0.53, 4.23) | (3.56, 0.57, 4.13) |
| 20v1 | (0.53, 3.70, 4.23) | (0.57, 3.56, 4.13) |
| 5v15 | (2.10, 1.03, 3.13) | (2.07, 1.06, 3.13) |
| 15v5 | (1.03, 2.10, 3.13) | (1.06, 2.07, 3.13) |

## Figures

![λ_h matrices](/opt/cursor/artifacts/rank_vs_rank_ipf_lambda_h.png)

![margin recovery](/opt/cursor/artifacts/rank_vs_rank_ipf_margins.png)

## Verdict

IPF enforces the bookkeeping identity (place GF/GA = sum of rank×rank λ). Match-level MAE vs actual scorelines is essentially unchanged vs the product map — the product was already nearly margin-consistent (±~2 season goals). Not promoted into cascade_long; park unless a player-GW G residual screen asks for it.
