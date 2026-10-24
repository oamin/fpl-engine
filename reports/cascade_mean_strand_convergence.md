# Generative mean-strand convergence vs centreline

Protocol: `cascade_mean_strand_convergence` · season **2025-26** · GW8–19 · Nmax=200 · seed=7

**Question:** as N→∞, does the average of generative strands approach the centreline (`score_team_side` on mid λ)?

## Answer (short)

**No — large N converges to the generative expectation, not the centreline.**  
Gen-mean vs centreline MAE falls from 2.15 (N=1) → **0.36 (N=200)** and is still ~0.36 across every GW — it plateaus above zero (corr≈0.955, small negative bias).  
Self-convergence continues: MAE(N=100 vs N=200)=**0.15**. Scoreline `k∼M` is nearly unbiased (Δk_h=+0.002); the residual gap is nonlinear player maps (CS / attack dice), matching prior viability (`gen clearly_loses` vs mean).

## Design

- Centreline: FD mid λ (GW8–9) / oracle-rank mid λ (GW10+) → `score_team_side` (no `k∼M`, no attack dice).
- Gen strand: same λ sources, but long weeks sample inside rankmean bands (`centre_blend=0.5`); then `k∼M` → attack dice (``lineup_cov=False``).
- One seed stream; prefixes reuse the same rolls.
- Wall: 338.3s

## Gen-mean(N) vs centreline

| N | MAE | RMSE | corr | bias (gen−centre) | max\|Δ\| |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.1460 | 2.7954 | 0.5167 | 0.0325 | 19.376 |
| 2 | 1.5475 | 2.0140 | 0.6487 | 0.0051 | 9.376 |
| 5 | 0.9942 | 1.3256 | 0.7832 | 0.0067 | 8.976 |
| 12 | 0.7025 | 0.9416 | 0.8759 | -0.0157 | 6.875 |
| 25 | 0.5433 | 0.7283 | 0.9185 | -0.0249 | 4.289 |
| 50 | 0.4431 | 0.6132 | 0.9407 | -0.0351 | 4.049 |
| 100 | 0.3908 | 0.5587 | 0.9509 | -0.0376 | 3.305 |
| 200 | 0.3627 | 0.5314 | 0.9548 | -0.0332 | 2.800 |

## Gen-mean(N) vs gen-mean(Nmax) (self-convergence)

| N | MAE | RMSE | corr |
| ---: | ---: | ---: | ---: |
| 1 | 2.0951 | 2.7553 | 0.5364 |
| 2 | 1.4743 | 1.9213 | 0.6879 |
| 5 | 0.9032 | 1.2016 | 0.8255 |
| 12 | 0.5822 | 0.7726 | 0.9182 |
| 25 | 0.3955 | 0.5241 | 0.9583 |
| 50 | 0.2490 | 0.3267 | 0.9833 |
| 100 | 0.1460 | 0.1963 | 0.9940 |
| 200 | 0.0000 | 0.0000 | 1.0000 |

## Scoreline moments at Nmax

- mean(k_h)−λ_h = **+0.0023** · mean(k_a)−λ_a = **-0.0086** (n_rolls=24000)
- If `k∼M` is calibrated, these biases → 0; player xP can still differ from centreline because CS/attack maps are nonlinear.

## Artefacts

- `data/processed/cascade_mean_strand_convergence/state.json`
- `reports/cascade_mean_strand_convergence.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_strand_convergence --season 2025-26 --n-max 200
```

