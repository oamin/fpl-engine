# Item 3 — Match-band calibration

Level uncertainty on Opta rank→GF/GA→λ bands for the multistage cascade.
Lead-time %/GW widening stays parked. Share seeding stays off.

## Pre-registered

- Baseline: `z_band=1.0`, `centre_blend=0.5`.
- Grid: `z ∈ [0.5, 1.0, 1.5, 2.0]`, `blend ∈ [0.0, 0.5, 1.0]`.
- Ranks: oracle final table; OLS / place profile LOSO on the other three closed seasons.
- Primary: eligible player-GW Gaussian log of MC-mean cascade xP (GW5–38, n_prior≥3, xmi≥45.0, σ=3.0).
- Secondary: season GF/GA residual coverage vs Gaussian nominal; MC strand-mean MAE; empirical 50/80% goal coverage; PIT.
- Adopt a non-baseline knob only if the pre-registered rule below fires.

## Decision

- Materiality bar (primary): mean player Δlog ≥ **+0.001** (grid max |Δ| = 0.00076).
- z closest to nominal coverage: **0.5** (Δlog=+0.00016, material=False)
- blend pick: **1.0** (cover_50=0.715, Δlog=+0.00021, material=False)
- **Change defaults: False** (adopt z=1.0, blend=0.5)

## Season residual coverage (mean over 4 LOSO seasons)

| z | mean cov | Gaussian nominal | |Δ| |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.294 | 0.383 | 0.089 |
| 1.0 | 0.512 | 0.683 | 0.170 |
| 1.5 | 0.756 | 0.866 | 0.110 |
| 2.0 | 0.837 | 0.954 | 0.117 |

## MC predictive (pooled means)

| z | blend | MAE strand | MAE λ | cover50 | cover80 | PIT mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.0 | 0.933 | 0.929 | 0.709 | 0.904 | 0.509 |
| 0.5 | 0.5 | 0.932 | 0.929 | 0.713 | 0.900 | 0.509 |
| 0.5 | 1.0 | 0.932 | 0.929 | 0.713 | 0.908 | 0.509 |
| 1.0 | 0.0 | 0.936 | 0.929 | 0.716 | 0.913 | 0.504 |
| 1.0 | 0.5 | 0.932 | 0.929 | 0.716 | 0.905 | 0.506 | ← baseline
| 1.0 | 1.0 | 0.933 | 0.929 | 0.715 | 0.912 | 0.508 |
| 1.5 | 0.0 | 0.943 | 0.929 | 0.725 | 0.921 | 0.499 |
| 1.5 | 0.5 | 0.938 | 0.929 | 0.723 | 0.909 | 0.502 |
| 1.5 | 1.0 | 0.934 | 0.929 | 0.717 | 0.912 | 0.507 |
| 2.0 | 0.0 | 0.949 | 0.929 | 0.736 | 0.926 | 0.494 |
| 2.0 | 0.5 | 0.942 | 0.929 | 0.724 | 0.919 | 0.497 |
| 2.0 | 1.0 | 0.935 | 0.929 | 0.719 | 0.915 | 0.505 |

## Player-GW log (MC-mean xP) — Δ vs baseline

Baseline mean log: **-2.52472**

| z | blend | mean log | Δ vs baseline |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.5 | -2.52456 | +0.00016 |
| 1.0 | 0.0 | -2.52508 | -0.00036 |
| 1.0 | 0.5 | -2.52472 | +0.00000 | ← baseline
| 1.0 | 1.0 | -2.52450 | +0.00021 |
| 1.5 | 0.5 | -2.52504 | -0.00032 |
| 2.0 | 0.5 | -2.52548 | -0.00076 |

## Read

1. Centreline match λ is unchanged by `z_band` — only MC width moves.
2. Season coverage is the proper check for the ±z·sd claim (goals-inside-λ-band mixes Poisson noise into a rate band).
3. Player-GW log barely moves across the grid (|Δ| ≪ materiality). MC cover_50 sits ~0.71 for every knob — discrete Poisson dominates band width.
4. Keep `Z_BAND=1.0` and `centre_blend=0.5`. Share seeding stays parked. Item 4 next: player G/A model.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.match_band_calibration
```
