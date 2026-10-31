# Cascade stepped xi_ft (π∝r) vs L3 / L4 (2025-26 GW8–12)

As of `2026-10-27T08:56:17Z` (UTC).

## Arms

- **cascade_step** — `cascade_mean_xi_ft_step_pool_resid` commits (full-club shares; goal_weight=`rate` — π∝r (minutes gate only); ≥60′ selection refreshed GW8…12; projected to GW19). Captain = Haaland if in XI. Seeded from `data/processed/l3_l4_gw08_seed_2025_26.json` (hold GW8: 0 xfers).
- **l3 / l4_mupost** — locked climb weeks from `data/processed/layer4_mupost_gamma1.json` (μ_post priced γ=1).

**Not gated.** Different boards (Opta generative long vs priced t:t+1).

![pts_net](/opt/cursor/artifacts/cascade_step_pi_r_vs_l3_l4.png)

## Totals

| arm | Σ pts_net | Σ XI+C | hits | transfers |
| --- | ---: | ---: | ---: | ---: |
| l4_mupost | 399 | 403 | 1 | 6 |
| l3 | 398 | 398 | 0 | 5 |
| cascade_step | 269 | 281 | 3 | 8 |

- cascade − L3: **-129** (-25.8/GW)
- cascade − L4: **-130** (-26.0/GW)

## Per GW (pts_net)

| GW | cascade | L3 | L4 | casc−L3 | casc−L4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 90 | 162 | 162 | -72 | -72 |
| 9 | 44 | 81 | 79 | -37 | -35 |
| 10 | 67 | 75 | 89 | -8 | -22 |
| 11 | 40 | 38 | 35 | +2 | +5 |
| 12 | 28 | 42 | 34 | -14 | -6 |

## Cascade squads (reminder)

- **GW8** C=Junior Kroupi · net=90
  - XI: David Raya Martín, Jurriën Timber, Junior Kroupi, Jaidon Anthony, Trevoh Chalobah, Ismaïla Sarr, Matheus Nunes, Erling Haaland, Daniel Ballard, Omar Alderete, Nick Woltemade
  - Bench: Ethan Ampadu, Chris Rigg, Guglielmo Vicario, Jhon Arias
- **GW9** C=Erling Haaland · net=44
  - XI: David Raya Martín, Gabriel dos Santos Magalhães, Jurriën Timber, Marcos Senesi Barón, Antoine Semenyo, Bryan Mbeumo, Ismaïla Sarr, Erling Haaland, Bruno Borges Fernandes, Omar Alderete, Nick Woltemade
  - Bench: Ethan Ampadu, Guglielmo Vicario, Junior Kroupi, Daniel Ballard
- **GW10** C=Erling Haaland · net=67
  - XI: David Raya Martín, Gabriel dos Santos Magalhães, Jurriën Timber, Bryan Mbeumo, Ismaïla Sarr, Dominik Szoboszlai, Erling Haaland, Bruno Borges Fernandes, Omar Alderete, Nick Woltemade, Daniel Ballard
  - Bench: Marcos Senesi Barón, Ethan Ampadu, Guglielmo Vicario, Junior Kroupi
- **GW11** C=Erling Haaland · net=40
  - XI: David Raya Martín, Gabriel dos Santos Magalhães, Jurriën Timber, Marcos Senesi Barón, Bryan Mbeumo, Ismaïla Sarr, Dominik Szoboszlai, Erling Haaland, Bruno Borges Fernandes, Junior Kroupi, Daniel Ballard
  - Bench: Ethan Ampadu, Guglielmo Vicario, Omar Alderete, Nick Woltemade
- **GW12** C=Erling Haaland · net=28
  - XI: David Raya Martín, Gabriel dos Santos Magalhães, Jurriën Timber, Marcos Senesi Barón, Bryan Mbeumo, Igor Thiago Nascimento Rodrigues, Ismaïla Sarr, Dominik Szoboszlai, Erling Haaland, Bruno Borges Fernandes, Junior Kroupi
  - Bench: Ethan Ampadu, Nico O'Reilly, Guglielmo Vicario, Daniel Ballard

## Verdict

On this window cascade stepped vs locked L3/L4: **-129** / **-130** net pts (GW8: 90 vs L3 162). GW9 took 5 transfers / 12 hit pts (net 44). GW8 uses the L3/L4 WC XI+captain; cascade pts are from `load_merged` actuals, while locked L3/L4 weeks keep the climb JSON totals (Kroupi GW8 is 12 merged vs 48 in `score_xp_v2_closed` — that alone is −72 on the same XI).

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_step_vs_l3_l4 --casc /opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_pi_r.json
```

