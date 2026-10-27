# cascade_long horizon residual — E[G, CS, GA] on t+2:end

Season **2025-26** · decision steps GW8–36 · boards **t+2…38** · N=40 · goal_weight=`rate` · match=`opta` · wall 3220.5s

## What this is

- Product cascade_long scope: Opta/oracle λ on **t+2 … end** (decision GW and t+1 excluded).
- Each step refreshes ≥60′ selection + full-club share pools.
- Shared L3 owned path; XI = WC seed or `pick_xi(score_xp_v2)`; **no captain**.
- Residual = Σ_XI (generative xP − actual P) on each target GW; horizon total = Σ over the plan.

![horizon shift](/opt/cursor/artifacts/cascade_long_horizon_component_resid_shift.png)

![target revision](/opt/cursor/artifacts/cascade_long_horizon_component_resid_revision.png)

## Horizon residual by decision GW

| t | n_plan | resid G | G/n | resid CS | CS/n | resid GA | GA/n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 29 | +90.9 | +3.14 | -10.0 | -0.34 | -16.3 | -0.56 |
| 9 | 28 | +40.3 | +1.44 | +13.8 | +0.49 | -8.8 | -0.31 |
| 10 | 27 | +102.9 | +3.81 | +43.3 | +1.60 | -13.2 | -0.49 |
| 11 | 26 | +83.6 | +3.22 | +32.0 | +1.23 | -11.5 | -0.44 |
| 12 | 25 | +82.1 | +3.29 | +18.3 | +0.73 | -16.6 | -0.66 |
| 13 | 24 | +150.6 | +6.27 | +38.9 | +1.62 | -17.0 | -0.71 |
| 14 | 23 | +106.0 | +4.61 | +22.5 | +0.98 | -33.0 | -1.43 |
| 15 | 22 | +59.5 | +2.70 | +53.9 | +2.45 | -12.3 | -0.56 |
| 16 | 21 | +65.4 | +3.11 | +51.6 | +2.46 | -6.6 | -0.31 |
| 17 | 20 | +48.8 | +2.44 | +24.4 | +1.22 | -12.6 | -0.63 |
| 18 | 19 | +73.0 | +3.84 | +26.3 | +1.38 | -12.5 | -0.66 |
| 19 | 18 | +73.0 | +4.06 | +11.1 | +0.62 | -7.8 | -0.43 |
| 20 | 17 | +83.7 | +4.92 | +36.3 | +2.14 | -7.0 | -0.41 |
| 21 | 16 | +91.8 | +5.74 | +28.9 | +1.80 | -9.8 | -0.61 |
| 22 | 15 | +69.8 | +4.65 | +9.9 | +0.66 | -10.4 | -0.69 |
| 23 | 14 | +74.8 | +5.34 | +32.7 | +2.33 | -11.7 | -0.84 |
| 24 | 13 | +61.2 | +4.71 | +26.8 | +2.06 | -6.3 | -0.48 |
| 25 | 12 | +54.9 | +4.58 | +20.1 | +1.68 | -3.5 | -0.29 |
| 26 | 11 | +28.2 | +2.57 | +15.4 | +1.40 | -5.1 | -0.46 |
| 27 | 10 | +35.7 | +3.57 | +7.9 | +0.79 | -2.8 | -0.28 |
| 28 | 9 | +36.4 | +4.05 | +9.8 | +1.09 | -2.1 | -0.24 |
| 29 | 8 | +23.1 | +2.89 | +13.6 | +1.70 | +1.3 | +0.16 |
| 30 | 7 | +11.8 | +1.68 | +10.4 | +1.48 | -0.2 | -0.03 |
| 31 | 6 | +9.4 | +1.56 | +7.7 | +1.28 | -6.2 | -1.03 |
| 32 | 5 | +12.3 | +2.47 | -5.5 | -1.11 | -4.9 | -0.97 |
| 33 | 4 | +13.3 | +3.33 | +3.7 | +0.93 | -4.3 | -1.08 |
| 34 | 3 | +7.8 | +2.60 | +5.8 | +1.94 | -1.5 | -0.49 |
| 35 | 2 | +10.5 | +5.24 | +6.2 | +3.12 | -1.0 | -0.50 |
| 36 | 1 | -1.1 | -1.10 | +7.5 | +7.47 | -1.1 | -1.05 |

## Verdict

- Horizon **Σ** resid_G shrinks as t advances mainly because `n_plan` shrinks; **per-GW** bias stays positive: mean cell resid_G **+3.7**, resid_CS **+1.3**, resid_GA **−0.6** (435 forecasts).
- Per decision, resid_G / n_plan stays roughly **+2…+6** through t≈28 (not a late-season calibration rescue).
- By offset: t+2…t+24 all show mean resid_G ≈ **+2…+4.5** — long Opta π∝r overprints goals at every horizon depth on this shared XI.
- CS horizon Σ swings −10…+54 early/mid then settles; per-GW mean mild overprint.
- Fixed-target revision paths are noisy because the L3 XI itself changes each step (actual P on the held XI moves).

## Mean horizon residual by decision block

| block | n | mean resid G | mean resid CS | mean resid GA | mean n_plan |
| --- | ---: | ---: | ---: | ---: | ---: |
| t=8-12 | 5 | +80.0 | +19.5 | -13.3 | 27.0 |
| t=13-19 | 7 | +82.3 | +32.7 | -14.5 | 21.0 |
| t=20-26 | 7 | +66.4 | +24.3 | -7.7 | 14.0 |
| t=27-33 | 7 | +20.3 | +6.8 | -2.7 | 7.0 |
| t=34-36 | 3 | +5.7 | +6.5 | -1.2 | 2.0 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_horizon_component_resid --n 40 --goal-weight rate --steps 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36 --horizon-end 38
```

Cells: `/opt/cursor/artifacts/cascade_long_horizon_component_resid_cells.json`

