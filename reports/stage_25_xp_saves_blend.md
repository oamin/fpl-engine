# Stage 25 — GKP saves + horizon xp/exp blend

Keeps linear **xP v2** as the core score. Adds deterministic GKP `xp_saves` and optional multi-GW blend `score_h = w_h·xp + (1−w_h)·exp_points` with `w_h = max(0.5, 0.85^h)` on the FT transfer V (lookahead H=5). XI/captain still use pure decision-GW score.

- Eval season: **2025-26**
- FT GWs: **4–38** (n=35)
- Default blend weights: h=0: 1.00, h=1: 0.85, h=2: 0.72, h=3: 0.61, h=4: 0.52

## GKP saves term

```
λ_conceded = clip(e_total − λ_scored, 0, 5)
E[saves]   = λ_conceded · 2.0     # empirical saves/GC
xp_saves   = p60 · E[saves]/3    # GKP only
```

- GKP mean `xp_saves`: **0.85**
- GKP level bias (pred−actual): **+0.13** (stage-24 was −0.72)

## Point-level quality

| universe | predictor | n | Spearman | MAE | bias |
|---|---|---:|---:|---:|---:|
| ALL | xP v2+saves | 9965 | 0.316 | 2.09 | -0.04 |
| ALL | exp points | 9965 | 0.257 | 2.16 | -0.03 |
| GKP | xP v2+saves | 655 | 0.176 | 2.17 | +0.13 |
| GKP | exp points | 655 | 0.034 | 2.22 | +0.12 |
| DEF | xP v2+saves | 3415 | 0.260 | 2.33 | +0.11 |
| DEF | exp points | 3415 | 0.172 | 2.40 | +0.01 |
| MID | xP v2+saves | 4647 | 0.373 | 1.92 | -0.09 |
| MID | exp points | 4647 | 0.320 | 1.98 | -0.06 |
| FWD | xP v2+saves | 1248 | 0.390 | 2.04 | -0.38 |
| FWD | exp points | 1248 | 0.342 | 2.17 | -0.09 |

## Stripped climb (captain ×2, no FT)

| method | total | mean/GW |
|---|---:|---:|
| xp | 2159 | 61.7 |
| blend_xp_exp | 2110 | 60.3 |
| exp_points | 1922 | 54.9 |
| roll3_points | 1797 | 51.3 |

## FT climb (H=5, autosubs + VC − hits)

Full ablation (γ / floor variants from the complete Stage 25 run):

| method | total | mean/GW | vs xp_ft | vs exp_ft |
|---|---:|---:|---:|---:|
| xp_blend90f4_ft (γ=0.9, floor=0.4) | 1995 | 57.0 | +8 | +107 |
| xp_ft | 1987 | 56.8 | +0 | +99 |
| xp_blend80_ft (γ=0.8, floor=0.5) | 1911 | 54.6 | −76 | +23 |
| exp_points_ft | 1888 | 53.9 | −99 | +0 |
| xp_blend85_ft (γ=0.85, floor=0.5) | 1842 | 52.6 | −145 | −46 |

## Gate verdicts

- **Saves:** PASS — GKP bias +0.13 (was −0.72)
- **Horizon blend:** WEAK — `xp_blend90f4` edges xp_ft by **+8** (pass bar ≥+10). Default γ=0.85 / floor=0.5 **hurts** (−145). Prefer pure xP; optional light blend only if re-validated.

## Plots / outputs

- `data/plots/stage_25_xp_saves_blend.png`
- `data/processed/stage_25_quality.csv`
- `data/processed/stage_25_ft_climb.csv`
