# Club expanding GF / GA — 2025/26 (seeded with 2024/25)

Four clubs: Arsenal, Liverpool, Man City, Chelsea.

Source: `data/cache/merged_gw_2025_26.csv`, one scored row per team–fixture, ordered by kickoff. Each club’s expanding window is **prepended with that club’s 38 scored matches from 2024/25** (`merged_gw_2024_25.csv`), so GW1 already sits on the prior season’s mean and dispersion instead of a 1-match spike.

Expanding mean, sample sd / variance, and median are match-order on prior + current. Dispersion is **sample variance / mean** (Fano factor). A Poisson process sits at 1. When the expanding mean is 0 the ratio is set to 0.

Series are plotted on GW1–38; blank weeks forward-fill the last expanding value (dotted verticals on the panel figures).

## 2024/25 prior (seed)

| Club | Mean GF | Disp GF | Mean GA | Disp GA |
| --- | ---: | ---: | ---: | ---: |
| Arsenal | 1.82 | 1.04 | 0.89 | 0.65 |
| Liverpool | 2.26 | 0.61 | 1.08 | 1.02 |
| Man City | 1.89 | 1.08 | 1.16 | 1.15 |
| Chelsea | 1.68 | 1.29 | 1.13 | 0.77 |

## Seeded expanding totals at GW38 (prior + 2025/26)

| Club | Mean GF | Median GF | Disp GF | Mean GA | Median GA | Disp GA | Blank GWs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Arsenal | 1.84 | 2.0 | 0.90 | 0.80 | 1.0 | 0.80 | 31 |
| Liverpool | 1.96 | 2.0 | 0.71 | 1.24 | 1.0 | 0.97 | — |
| Man City | 1.96 | 2.0 | 0.93 | 1.04 | 1.0 | 1.11 | 31, 34 |
| Chelsea | 1.61 | 1.0 | 1.10 | 1.25 | 1.0 | 0.77 | 34 |

## Figures

- `reports/figures/club_expanding_gf_ga_2025_26.png` — per club: expanding mean ± 1 sd band and expanding median for GF and GA.
- `reports/figures/club_expanding_mean_overview_2025_26.png` — four-club expanding mean GF and GA.
- `reports/figures/club_expanding_dispersion_2025_26.png` — four-club expanding dispersion (var/mean) for GF and GA.
- `reports/figures/club_expanding_dispersion_panel_2025_26.png` — per-club dispersion panels.
- `reports/figures/club_expanding_gf_ga_2025_26.json` — series used by the plots.

Reproduce: `python3 scripts/plot_club_expanding_goals.py`.
