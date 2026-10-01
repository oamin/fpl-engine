# Stage 13 — Thin xP engine + Spearman IC gate

Deterministic component xP from **xMi + market λ/CS proxy + expanding shares + DefCon hit rate**. No sims, no Odds API props.

- Joined player-match rows: **11498**
- Gate: Spearman IC vs **mean points over next H GWs**
- Baselines: exp points, roll3 points, xMi, price

## Formula (v1)

```
appear  ≈ 2 if xMi≥60 else 1+xMi/60 (if xMi>0)
λ       ≈ e_total(OU) × attack_strength / (att+threat)
P(CS)   ≈ clip(0.08 + 0.35·p_not_lose + 0.15·p_under)
xP      = appear
        + play_scale · share_xG · λ · goal_pts
        + play_scale · share_xA · λ_a · 3
        + P60 · P(CS) · cs_pts
        + P60 · P(DefCon hit) · 2
```

## IC — regulars xMi≥45

| predictor | H=1 | H=3 | H=5 | H=8 |
|---|---:|---:|---:|---:|
| xP engine | 0.159 | 0.215 | 0.254 | 0.296 |
| exp points | 0.155 | 0.237 | 0.287 | 0.341 |
| roll3 points | 0.110 | 0.139 | 0.155 | 0.202 |
| xMi | 0.130 | 0.178 | 0.204 | 0.231 |
| price (value) | 0.117 | 0.153 | 0.190 | 0.229 |
| exp xG only | 0.070 | 0.106 | 0.139 | 0.175 |
| exp DefCon hit only | 0.074 | 0.117 | 0.139 | 0.156 |

## IC — ALL appearances

| predictor | H=1 | H=3 | H=5 | H=8 |
|---|---:|---:|---:|---:|
| xP engine | 0.259 | 0.314 | 0.346 | 0.378 |
| exp points | 0.237 | 0.310 | 0.350 | 0.397 |
| xMi | 0.247 | 0.294 | 0.318 | 0.339 |
| price (value) | 0.108 | 0.133 | 0.154 | 0.187 |

## xP vs exp points by position (regulars xMi≥45, H=5)

| pos | xP Spearman | exp pts Spearman | Δ |
|---|---:|---:|---:|
| GKP | 0.128 | -0.075 | +0.203 |
| DEF | 0.307 | 0.287 | +0.019 |
| MID | 0.346 | 0.332 | +0.015 |
| FWD | 0.312 | 0.283 | +0.030 |

## Gate verdict

**FAIL — xP trails exp points by -0.044 IC at H=8**

Pass bar: xP Spearman IC ≥ exp points at H=5 and H=8 on regulars (prefer Δ ≥ 0.03 before investing in MILP).

## Plots

- `data/plots/xp_ic_gate.png`

## Output

- `data/processed/xp_engine.csv` — per player-match xP + components
- `data/processed/xp_ic_gate.csv`
