# Form pattern in Opta−FD residual (2025-26)

Pre-registered **2026-10-29** · as-of form (before GW) vs residual `Opta λ − FD λ` · GW8–38 · n=518 team-sides with ≥5 prior games

## Verdict

- **Yes, after controlling for place quality.** Raw L5 form barely moves the residual (overall hot−cold ≈ 0) because form is confounded with finishing rank — and strong clubs already carry a positive Opta−FD gap.
- Conditional on rank + home/away: partial corr(L5 GD, Opta−FD) = **-0.229**.
- **+1 SD** recent GD → Opta−FD **-0.091** (FD higher vs Opta by **0.091**).
- FD responds to form (partial r = **+0.239**); Opta ranking λ almost does not (**+0.025**) — place-locked, as designed.
- Within top and bottom rank bands, in-form vs out-of-form residual gap is about **−0.13** (FD swung up when hot).

## Why raw tertiles look flat

Hot L5 GD teams are better clubs; Opta place λ is already high on them, so mean Opta−FD stays positive. The form effect only appears once rank (and venue) are held fixed.

## Within finishing-rank band (L5 GD tertiles)

| rank band | cold resid | hot resid | hot−cold |
| --- | ---: | ---: | ---: |
| top | +0.394 | +0.261 | -0.133 |
| mid | -0.079 | -0.049 | +0.030 |
| bottom | -0.009 | -0.142 | -0.133 |

R²(rank + home) = 0.296; with L5 GD form = 0.332 (Δ = +0.037).

## Reproduce

Rows: `data/processed/cascade_long_fd_opta_form_resid_rows.csv`

