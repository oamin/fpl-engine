# Stage 23 — Horizon uncertainty + FDR + transfer-flow

Transfer score at horizon step `h`:

```text
sc = μ · n_fixtures · w_σ(h) · fdr(gw_h) · flow(t)
σ_h = σ0 · √(1+β·h);  w_σ = τ²/(τ²+σ_h²)
β=1.0, τ=3.0, flow κ=0.06, H=4
```

- **σ0:** expanding std of `(points − score_xp)`, prior-only
- **fdr:** MID/FWD via `attack_strength`; GKP/DEF via `1−defend_threat`
- **flow:** decision-GW `transfers_balance` within-GW z → clip ±15%

- GWs: **5–38** (n=34)
- Stage-22 ridge_h3 baseline: **1872**

## Final standings (captain ×2 − hits)

| method | total | mean/GW | vs full | vs stage22 |
|---|---:|---:|---:|---:|
| xp_budget | 2116 | 62.2 | +365 | — |
| ridge_h3_fdr_ft | 1892 | 55.6 | +141 | +20 |
| ridge_h3_nounc_ft | 1872 | 55.1 | +121 | +0 |
| xp_ft | 1804 | 53.1 | +53 | — |
| ridge_h3_flow_ft | 1768 | 52.0 | +17 | -104 |
| ridge_h3_ft | 1751 | 51.5 | +0 | -121 |

## Transfer behaviour

- ridge_h3_ft (full): tx=1.00, holds=8/34, hits=3
- ridge_h3_nounc_ft: tx=0.88, holds=19/34, hits=0
- ridge_h3_fdr_ft: tx=0.97, holds=13/34, hits=0
- ridge_h3_flow_ft: tx=0.94, holds=9/34, hits=1
- xp_ft: tx=1.06, holds=4/34, hits=4

## Ablations

- Full vs stage22 nounc: **-121**
- FDR-only vs nounc: **+20**
- Flow-only vs nounc: **-104**
- Full vs xp_ft: **-53**
- Best FT vs xp_budget (2116): **-224**

## Mean weights by horizon (ridge_h3 full squad)

| h | mean w_σ | mean fdr | mean flow |
|---:|---:|---:|---:|
| 0 | 0.546 | 1.056 | 1.008 |
| 1 | 0.396 | 1.043 | 1.009 |
| 2 | 0.317 | 1.045 | 1.008 |
| 3 | 0.268 | 1.046 | 1.008 |

## Gate verdict

**PARTIAL — component helps (fdr) but full stack -121 vs nounc**

## Plots

- `data/plots/season_climb_stage23.png`

## Output

- `data/processed/season_climb_stage23.csv`
