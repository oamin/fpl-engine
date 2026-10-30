# L3@t + Opta λ horizon — vs contemporaneous L3@g + actuals

Season **2025-26** · decisions 31 · horizon_end=38 · n=89444 player-GW cells · wall 116.7s

Pre-registered: `protocol.l3_opta_lambda_horizon`.

## Intent

Alternative to cascade generative short/long for strategy scoring:
keep **L3/L4 centreline mechanics** (`score_team_side`), but drive far (and near) fixtures with **Opta centreline λ** from a freeze at decision GW `t`, forecasting every `g ∈ [t, end]`.

## Arms

- **L3@t + Opta λ** — rates/shares/alphas frozen at `t`; Opta blind place λ → DC moments → `score_team_side`.
- **L3@g + FD λ** — contemporaneous centreline (FD priced near `g`).
- **Actuals** — `P_G` / `P_CS` / `P_GC` on `minutes>60`.

![offset mae](/opt/cursor/artifacts/l3_opta_lambda_horizon_offset_mae.png)

## Overall

| arm | G bias | G MAE | CS bias | CS MAE | GC bias | GC MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L3@t + Opta λ | -0.090 | 0.818 | -0.204 | 0.758 | -0.059 | 0.252 |
| L3@g + FD λ | -0.111 | 0.805 | -0.192 | 0.758 | -0.047 | 0.248 |

## Near vs far

| band | n | G ΔMAE (Opta−L3) | CS ΔMAE | GC ΔMAE |
| --- | ---: | ---: | ---: | ---: |
| near_0_1 | 11721 | +0.013 | -0.006 | +0.001 |
| far_2plus | 77723 | +0.012 | +0.000 | +0.004 |

## G MAE by horizon offset

| offset | n | Opta MAE | L3 MAE | Δ (Opta−L3) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5986 | 0.821 | 0.806 | +0.015 |
| 1 | 5735 | 0.824 | 0.812 | +0.012 |
| 2 | 5495 | 0.822 | 0.813 | +0.010 |
| 3 | 5255 | 0.820 | 0.812 | +0.009 |
| 4 | 5022 | 0.820 | 0.812 | +0.008 |
| 5 | 4803 | 0.818 | 0.810 | +0.008 |
| 6 | 4592 | 0.812 | 0.805 | +0.007 |
| 7 | 4383 | 0.813 | 0.805 | +0.008 |
| 8 | 4173 | 0.812 | 0.802 | +0.010 |
| 9 | 3957 | 0.809 | 0.797 | +0.011 |
| 10 | 3755 | 0.812 | 0.800 | +0.012 |
| 11 | 3567 | 0.813 | 0.800 | +0.013 |
| 12 | 3370 | 0.817 | 0.802 | +0.015 |
| 13 | 3180 | 0.811 | 0.794 | +0.017 |
| 14 | 2989 | 0.807 | 0.790 | +0.017 |
| 15 | 2792 | 0.812 | 0.795 | +0.017 |
| 16 | 2610 | 0.813 | 0.795 | +0.019 |
| 17 | 2426 | 0.820 | 0.800 | +0.020 |
| 18 | 2244 | 0.826 | 0.805 | +0.021 |
| 19 | 2051 | 0.815 | 0.798 | +0.017 |
| 20 | 1874 | 0.822 | 0.805 | +0.017 |
| 21 | 1688 | 0.823 | 0.804 | +0.020 |
| 22 | 1516 | 0.821 | 0.801 | +0.019 |
| 23 | 1344 | 0.841 | 0.822 | +0.019 |
| 24 | 1197 | 0.841 | 0.820 | +0.021 |
| 25 | 1020 | 0.837 | 0.817 | +0.021 |
| 26 | 788 | 0.823 | 0.826 | -0.003 |
| 27 | 666 | 0.833 | 0.840 | -0.007 |
| 28 | 499 | 0.831 | 0.845 | -0.014 |
| 29 | 316 | 0.839 | 0.851 | -0.012 |
| 30 | 151 | 0.731 | 0.743 | -0.012 |

## Verdict

- Near G ΔMAE **+0.013**; far **+0.012** (positive ⇒ L3@g closer to actuals).
- Overall G MAE: L3@t+Opta **0.818** vs L3@g+FD **0.805**.
- **vs generative cascade_long @t** (prior dig, G MAE ≈0.846; near ΔMAE +0.054):
  centreline+Opta is clearly better — most of the earlier miss was
  **generative dice / MC-mean**, not Opta λ itself.
- CS: near ΔMAE **−0.006** (slightly beats L3@g); far flat.
- Feasibility: **promising** as a strategy scorer (L3 mechanics + Opta λ
  for the unpriced horizon). Not a default flip yet.
- Defaults unchanged.

## Reproduce

```
PYTHONPATH=. python3 -m src.eval.l3_opta_lambda_horizon --horizon-end 38
```

