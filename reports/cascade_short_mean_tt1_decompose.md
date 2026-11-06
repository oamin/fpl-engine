# Where the t:t+1 gen−mean Δ comes from

Screen: `cascade_short_mean_tt1_vs_gen` (N_MC=20). Overall Δlog ≈ **−0.054**.

## Not the horizon

| Slice | Δlog |
| --- | --- |
| score week = t | −0.051 |
| score week = t+1 | −0.057 |

Same gap on both legs — not “mean only looks at t.”

## Position (the lever)

| Pos | n | Δlog | Share of loss mass |
| --- | --- | --- | --- |
| MID | 12.9k | **−0.095** | ~76% |
| FWD | 2.9k | **−0.163** | ~29% |
| DEF | 11.3k | +0.006 | (slight help) |
| GKP | 2.9k | +0.005 | (slight help) |

**FWD+MID ≈ 105% of the total negative mass** (DEF/GKP offset a little).

## Extremes / calibration

- Gen forecasts are **wider** (std 1.90 vs mean 1.57).
- Top gen decile: pred **6.81** vs actual **5.06** (overshoot −1.75); mean top decile nearly calibrated (5.63 vs 5.38).
- High-μ tertile Δlog **−0.121** (~75% of loss mass); low-μ tertile gen **wins** (+0.031).
- Rows with \|gen−mean\|≥1: 29% of rows, **85%** of loss mass.
- Gen **colder** than mean by ≥1: Δlog **−0.223** (61% of loss) — lineup/attack dice often zero out attackers vs closed-form μ.
- Gen **hotter** by ≥1: Δlog **−0.090** — when gen spikes premiums it also overshoots.

## Read

The delta is mostly **attack-path discreteness**: generative lineup + goal dice make MID/FWD forecasts noisier and worse-calibrated at the top, while DEF/GKP (CS-driven) are roughly fine. It is not a t:t+1 framing artifact.
