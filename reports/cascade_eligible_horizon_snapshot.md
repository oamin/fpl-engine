# Eligible-pool horizon snapshots — cascade @t vs L3 @g vs actuals

Season **2025-26** · decisions 31 · horizon_end=38 · N=20 · goal_weight=`rate_minutes` · n=85601 player-GW cells · short cells=11387 · wall 2385.9s

Pre-registered: `protocol.cascade_eligible_horizon_snapshot`.

## Intent (correct design)

At decision GW **t**, freeze as-of state and forecast **every target** `g ∈ [t, horizon_end]` with cascade mechanics. Compare each cell to:

- **L3 @g** — contemporaneous `score_xp_v2` using FD priced near `g`
  (should be the sharper near-term predictor), and
- **actuals** at `g` (residual only if `minutes>60`).

Then step `t → t+1` and repeat. This supersedes the decision-GW-only `cascade_eligible_component_resid` dig.

## Arms

- **cascade_long @t** — Opta blind place λ → generative MC-mean for all `g`.
- **cascade_short @t** — FD λ only for `g ∈ {t, t+1}` (no future-FD leak).
- **L3 @g** — as-of before `g` + that GW's FD centreline components.

![offset mae](/opt/cursor/artifacts/cascade_eligible_horizon_snapshot_offset_mae.png)

## Overall (all freeze×target cells)

| arm | G bias | G MAE | CS bias | CS MAE | GC bias | GC MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long Opta @t | -0.062 | 0.846 | -0.128 | 0.775 | -0.052 | 0.247 |
| L3 FD @g | -0.114 | 0.820 | -0.174 | 0.750 | -0.044 | 0.243 |

Short cells only (offset ≤1, n≈11387):

| arm | G MAE | CS MAE | GC MAE |
| --- | ---: | ---: | ---: |
| short FD @t | 0.859 | 0.796 | 0.245 |
| L3 FD @g (same rows) | 0.814 | 0.749 | 0.242 |

## Near vs far (long Opta @t vs L3 @g)

| band | n | G ΔMAE (long−L3) | CS ΔMAE | GC ΔMAE |
| --- | ---: | ---: | ---: | ---: |
| near_0_1 | 11387 | +0.054 | +0.038 | +0.004 |
| far_2plus | 74214 | +0.021 | +0.023 | +0.004 |

## G MAE by horizon offset

| offset | n | long MAE | L3 MAE | Δ (long−L3) | short MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5892 | 0.877 | 0.814 | +0.063 | 0.864 |
| 1 | 5495 | 0.861 | 0.816 | +0.045 | 0.855 |
| 2 | 5209 | 0.851 | 0.815 | +0.036 | — |
| 3 | 4984 | 0.864 | 0.827 | +0.037 | — |
| 4 | 4726 | 0.854 | 0.822 | +0.032 | — |
| 5 | 4525 | 0.835 | 0.820 | +0.014 | — |
| 6 | 4322 | 0.858 | 0.823 | +0.035 | — |
| 7 | 4119 | 0.837 | 0.820 | +0.017 | — |
| 8 | 3944 | 0.840 | 0.816 | +0.024 | — |
| 9 | 3760 | 0.832 | 0.810 | +0.022 | — |
| 10 | 3577 | 0.835 | 0.813 | +0.022 | — |
| 11 | 3397 | 0.838 | 0.813 | +0.025 | — |
| 12 | 3220 | 0.840 | 0.817 | +0.023 | — |
| 13 | 3030 | 0.835 | 0.813 | +0.022 | — |
| 14 | 2846 | 0.834 | 0.807 | +0.027 | — |
| 15 | 2645 | 0.838 | 0.820 | +0.017 | — |
| 16 | 2473 | 0.835 | 0.810 | +0.026 | — |
| 17 | 2322 | 0.829 | 0.816 | +0.013 | — |
| 18 | 2165 | 0.842 | 0.824 | +0.018 | — |
| 19 | 1995 | 0.828 | 0.826 | +0.003 | — |
| 20 | 1830 | 0.822 | 0.826 | -0.004 | — |
| 21 | 1649 | 0.845 | 0.833 | +0.012 | — |
| 22 | 1491 | 0.831 | 0.831 | -0.000 | — |
| 23 | 1328 | 0.874 | 0.859 | +0.015 | — |
| 24 | 1191 | 0.877 | 0.856 | +0.021 | — |
| 25 | 1023 | 0.877 | 0.844 | +0.033 | — |
| 26 | 789 | 0.823 | 0.846 | -0.023 | — |
| 27 | 670 | 0.821 | 0.840 | -0.019 | — |
| 28 | 502 | 0.812 | 0.842 | -0.030 | — |
| 29 | 324 | 0.846 | 0.858 | -0.012 | — |
| 30 | 158 | 0.686 | 0.740 | -0.054 | — |

## Verdict

- Far (offset≥2) long−L3 G ΔMAE **+0.021**; near (0–1) **+0.054**.
- Overall long G MAE 0.846 vs L3 0.820.
- Short vs L3 on causal t:t+1 cells: G MAE 0.859 vs 0.814.
- Defaults unchanged (diagnostic).

## Reproduce

```
PYTHONPATH=. python3 -m src.eval.cascade_eligible_horizon_snapshot --n 20 --horizon-end 38
```

