# Stage 28 — Sharpe-u free-transfer climb

Agreed with the Co-PI before the run. The score is
\(u = \mu / (\sigma + 1)\), \(\mu =\) `score_xp`.
\(\sigma\) is the expanding sample std of `total_points − score_xp`
after a one-gameweek shift, needing 3 prior residuals.
A missing sigma uses the position median of earlier gameweeks only, else 3.0,
then clipped to [0.5, 8]. That u is frozen for the whole transfer horizon.

Rules held fixed from `season_climb_ft`: no chips, H=3, hold unless ΔV ≥ 1.25, hit −4, formations unchanged.
Gameweeks **5–38** (n=34), 2025-26.
The comparator is the paired `xp_ft` from this same run, not an older published total.

## Standings (captain ×2 − hits)

| method | total | mean/GW | vs xp_ft |
|---|---:|---:|---:|
| xp_ft | 1868 | 54.9 | +0 |
| sharpe_u_ft | 1615 | 47.5 | -253 |

## Transfers (the disagreeing diagnostic)

| method | mean transfers/GW | hits | hit points |
|---|---:|---:|---:|
| xp_ft | 1.03 | 3 | 12 |
| sharpe_u_ft | 0.85 | 0 | 0 |

If the total rises while hits and transfers both rise, treat it as churn, not a risk edge.

## Captain on the same XI

On the XI picked by u, argmax u and argmax mu differed in **32** / 34 gameweeks.
Played-minutes points of the mu captain minus the u captain: **+43**.
Positive means the expected-points captain scored more on that same XI.
This does not re-optimise transfers, and it does not apply the vice-captain.

## Gate

**FAIL — sharpe_u_ft vs paired xp_ft is -253 (bar +34). Park the objective.**

The Co-PI agreed. Hits and transfers both fell, so the loss is not churn. Captain choice on the same XI explains about 43 of the 253. Replacing expected points with u = mu / (sigma + 1) is rejected. A milder penalty, or using sigma only inside the transfer horizon, is not ruled out.

Pre-registered bar: +34 versus the paired xp_ft. The bar was not moved after the result.

## Output

- `data/processed/stage_28_sharpe_u.csv`
- `data/plots/stage_28_sharpe_u.png`
