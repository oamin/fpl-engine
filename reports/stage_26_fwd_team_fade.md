# Stage 26 — FWD goal calibration + team-prior horizon fade

Keeps linear **xP v2** as the core score. Adds:
1. Leakage-free **FWD goal scale** = expanding `mean(goals)/mean(e_goals)` (clipped), applied to `xp_goals` before BPS.
2. Optional FT-V blend toward **team-strength prior** (expanding team λ/CS), with `w(h)=1` for h≤3, fading to 0.60 at h=5.

- Eval: **2025-26**, FT GWs **4–38** (n=35)
- Team-fade weights: h=0: 1.00, h=1: 1.00, h=2: 1.00, h=3: 1.00, h=4: 0.80
- Stage-25 pure xP FT baseline: **1987**

## FWD calibration

- Mean FWD `fwd_goal_scale`: **1.060**
- FWD bias: **-0.10** (stage-25 was −0.38; target ∈ [−0.10, +0.10])

## Point-level quality

| universe | predictor | n | Spearman | MAE | bias |
|---|---|---:|---:|---:|---:|
| ALL | xP v2+FWD cal | 9965 | 0.317 | 2.11 | -0.01 |
| ALL | exp points | 9965 | 0.257 | 2.16 | -0.03 |
| GKP | xP v2+FWD cal | 655 | 0.176 | 2.17 | +0.13 |
| GKP | exp points | 655 | 0.034 | 2.22 | +0.12 |
| DEF | xP v2+FWD cal | 3415 | 0.260 | 2.33 | +0.11 |
| DEF | exp points | 3415 | 0.172 | 2.40 | +0.01 |
| MID | xP v2+FWD cal | 4647 | 0.373 | 1.92 | -0.09 |
| MID | exp points | 4647 | 0.320 | 1.98 | -0.06 |
| FWD | xP v2+FWD cal | 1248 | 0.385 | 2.14 | -0.10 |
| FWD | exp points | 1248 | 0.342 | 2.17 | -0.09 |

## Stripped climb

| method | total | mean/GW |
|---|---:|---:|
| xp | 2185 | 62.4 |
| blend_xp_exp | 2084 | 59.5 |
| team_prior | 2042 | 58.3 |
| exp_points | 1925 | 55.0 |

## FT climb (H=5)

| method | total | mean/GW | vs xp_ft | vs stage25 baseline |
|---|---:|---:|---:|---:|
| xp_ft | 1918 | 54.8 | +0 | -69 |
| xp_team_fade_ft | 1901 | 54.3 | -17 | -86 |
| exp_points_ft | 1888 | 53.9 | -30 | -99 |

## Transfer stability

| method | total transfers | mean/GW | hits |
|---|---:|---:|---:|
| exp_points_ft | 25 | 0.71 | 0 |
| xp_ft | 48 | 1.37 | 15 |
| xp_team_fade_ft | 46 | 1.31 | 13 |

- Team-fade / pure-xP transfer ratio: **0.958** (cap ≤ 1.05)

## Gate verdicts

- **FWD bias:** PASS — FWD bias −0.10 (was −0.38; at the ±0.10 edge)
- **FT climb:** FAIL — best FT 1918 trails stage-25 baseline by −69 (more hits: 15 vs prior path)
- **Transfers:** PASS — team-fade transfer ratio 0.96 (no chitter)

**Read:** FWD calibration fixes level bias and lifts the **stripped** climb (2185 vs ~2159). The same additive level term **hurts FT** (−69 vs 1987) via extra hits. Team-prior fade is transfer-stable but does not beat pure xP. Prefer **goal-scale on**, treat **level-add as optional** for ranking-only contexts; leave team-fade off by default.

## Outputs

- `data/plots/stage_26_fwd_team_fade.png`
- `data/processed/stage_26_quality.csv`
- `data/processed/stage_26_ft_climb.csv`
