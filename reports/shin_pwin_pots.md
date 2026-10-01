# Shin P(win) vs team×pos pots

Shin P(win) = side-specific win probability from Shin-devigged 1X2 (home uses `p_home`, away uses `p_away`).

## Match-level

| pos | n | slope β | corr | R² |
|---|---:|---:|---:|---:|
| GKP | 760 | 2.43 | 0.166 | 0.028 |
| DEF | 760 | 14.68 | 0.246 | 0.060 |
| MID | 760 | 17.91 | 0.385 | 0.149 |
| FWD | 748 | 1.18 | 0.049 | 0.002 |

## Decile calibration

| pos | n | slope β | match R² | decile_r |
|---|---:|---:|---:|---:|
| GKP | 760 | 2.43 | 0.028 | 0.836 |
| DEF | 760 | 14.68 | 0.060 | 0.913 |
| MID | 760 | 17.91 | 0.149 | 0.980 |
| FWD | 748 | 1.18 | 0.002 | 0.437 |

## Plots

- `data/plots/shin_pwin_vs_pot.png` — match scatter
- `data/plots/shin_pwin_vs_pot_deciles.png` — decile means
- `data/plots/shin_pwin_vs_wdl_compare.png` — WDL bars vs P(win) curves

## Read

- Use **Shin P(win)** as the pre-match fixture feature (no outcome leakage).
- DEF/MID show clear positive slopes; FWD is nearly flat.
- Decile curves are the fair visual for “does the market move the pot?”
