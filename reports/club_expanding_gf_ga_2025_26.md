# Club expanding GF / GA — 2025/26

Four clubs: Arsenal, Liverpool, Man City, Chelsea.

Source: `data/cache/merged_gw_2025_26.csv`, one scored row per team–fixture, ordered by kickoff. Expanding mean, sample sd / variance, and median are match-order. Dispersion is the index of dispersion: **sample variance / mean** (Fano factor). A Poisson process sits at 1; below 1 is underdispersed, above 1 overdispersed. When the expanding mean is 0 the ratio is set to 0.

Series are plotted on GW1–38; blank weeks forward-fill the last expanding value (dotted verticals on the panel figures).

## Season totals (38 matches)

| Club | Mean GF | SD GF | Median GF | Disp GF | Mean GA | SD GA | Median GA | Disp GA | Blank GWs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Arsenal | 1.87 | 1.21 | 2.0 | 0.79 | 0.71 | 0.84 | 0.5 | 0.98 | 31 |
| Liverpool | 1.66 | 1.12 | 1.5 | 0.76 | 1.39 | 1.13 | 1.0 | 0.91 | — |
| Man City | 2.03 | 1.28 | 2.0 | 0.81 | 0.92 | 1.00 | 1.0 | 1.08 | 31, 34 |
| Chelsea | 1.53 | 1.18 | 1.0 | 0.91 | 1.37 | 1.03 | 1.0 | 0.77 | 34 |

All four clubs finish underdispersed on GF (var/mean &lt; 1). Man City alone finishes slightly overdispersed on GA (1.08).

## Figures

- `reports/figures/club_expanding_gf_ga_2025_26.png` — per club: expanding mean ± 1 sd band and expanding median for GF and GA.
- `reports/figures/club_expanding_mean_overview_2025_26.png` — four-club expanding mean GF and GA.
- `reports/figures/club_expanding_dispersion_2025_26.png` — four-club expanding dispersion (var/mean) for GF and GA.
- `reports/figures/club_expanding_dispersion_panel_2025_26.png` — per-club dispersion panels.
- `reports/figures/club_expanding_gf_ga_2025_26.json` — series used by the plots.

Reproduce: `python3 scripts/plot_club_expanding_goals.py`.
