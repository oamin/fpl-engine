# Correlated draws — calibration vs realised co-movement

## 1. σ_log_lam from football-data xG vs goals

- Sides: **3038** (four closed seasons)
- Var(goals − xG): **1.375** vs Poisson E[xG]=**1.497**
- Excess var: **-0.122** → overdispersion σ_λ ≈ **0.000**
- SD of log((G+0.5)/(xG+0.5)): **0.689** (upper bound; includes outcome noise)

Overdispersion σ is 0 when Var(G−xG) ≤ E[xG] (common here). Log-residual SD mixes outcome noise with λ error — upper bound only. Player co-movement needs σ_idio; match-shock alone is not residual-scale.

## 2. Realised player residual co-movement

resid = total_points − score_xp_v2; minutes≥60; unique element×fixture; attackers=MID/FWD

- Same-team MID/FWD resid corr: **+0.068** (n=31984)
- Attacker vs opposing GKP resid corr: **-0.276** (n=14131)
- Residual SD (minutes≥60): **3.12** pts

| season | atk–atk | atk–opp GK |
| --- | ---: | ---: |
| 2022-23 | +0.103 | -0.316 |
| 2023-24 | +0.076 | -0.258 |
| 2024-25 | +0.059 | -0.269 |
| 2025-26 | +0.032 | -0.263 |

## 3. Heterogeneous σ_out + match σ_λ (toy fixture)

Idio = position mean **sigma_out** (expanding residual SD; see `reports/outcome_spread.md`). **Not** `xp_se`. Structural path = λ shocks → scoreline → player xP; then N(0, σ_out,j²) per player.

Idio = position mean sigma_out (expanding residual SD). xp_se is not used. Marginal draw SD ≈ sigma_out; coupling from σ_λ.

- Position σ_out priors: DEF=2.91, FWD=3.58, GKP=2.66, MID=2.94
- Chosen **σ_log_lam = 0.300**
- Draw corr home mids: **+0.311** (target +0.068)
- Draw corr home M1 vs away GK: **-0.062** (target -0.276)

### Sweep

| σ_log_lam | mid–mid xP | M1 vs away GK | loss |
| ---: | ---: | ---: | ---: |
| 0.050 | +0.063 | +0.085 | 0.2614 |
| 0.080 | +0.078 | +0.078 | 0.2510 |
| 0.100 | +0.092 | +0.071 | 0.2414 |
| 0.120 | +0.107 | +0.062 | 0.2304 |
| 0.150 | +0.135 | +0.046 | 0.2123 |
| 0.200 | +0.190 | +0.013 | 0.1830 |
| 0.300 | +0.311 | -0.062 | 0.1512 ← |
| 0.400 | +0.416 | -0.136 | 0.1608 |
| 0.500 | +0.494 | -0.201 | 0.1935 |

## Verdict

Ship **σ_log_lam=0.300** with per-player **σ_out** (position means MID=2.94, GKP=2.66). Draw mid–mid +0.311 vs realised +0.068 (|Δ|=0.244); opp-GK -0.062 vs -0.276 (|Δ|=0.214). Scalar fallback σ_idio=2.94. xp_se stays shrink-only.

Defaults: `data/processed/correlated_draws_defaults.json`
JSON: `data/processed/correlated_draws_calibration.json`
σ_out table: `data/processed/outcome_spread_closed.csv`

