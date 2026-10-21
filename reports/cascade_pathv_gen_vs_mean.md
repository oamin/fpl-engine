# Path-V diagnostic — generative vs mean-layer FT+captain

Protocol: `cascade_pathv_gen_vs_mean_milp` · N=20 · seed=7 · decision GW6 → horizon 19 · priced GW8 · objective `xi_plus_captain_ft`

**Diagnostic only** — not a gated four-season path-V promote.

- Smoke passed: **True**
- Top-15 freq Jaccard (gen ∩ mean): **0.364**
- Wall: 1065.6s

## Generative strands

- Squads: 120 · strands_with_squads=20
- Mean transfers / hits: 3.79 / 7.16
- Sources: ['cascade_long_opta', 'cascade_short_priced']
- Scoring: GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→attack→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW
- Top captain: {'name': 'Groß', 'freq_captain': 0.15}
- Elapsed: 599.4s

Top freq:

- Tzolakis (hull): freq=0.9333333333333333 xi=0.14166666666666666 cap=0.0
- Groß (brighton): freq=0.9166666666666666 xi=0.9083333333333333 cap=0.15
- Tarkowski (everton): freq=0.9083333333333333 xi=0.875 cap=0.15
- Dasilva (coventry): freq=0.8666666666666667 xi=0.7 cap=0.09166666666666666
- Hall (newcastle): freq=0.8666666666666667 xi=0.775 cap=0.13333333333333333
- Van Hecke (spurs): freq=0.825 xi=0.6666666666666666 cap=0.125
- Wirtz (liverpool): freq=0.6583333333333333 xi=0.6166666666666667 cap=0.058333333333333334
- Hume (sunderland): freq=0.525 xi=0.0 cap=0.0
- Belloumi (hull): freq=0.5083333333333333 xi=0.43333333333333335 cap=0.0
- N.Jackson (aston villa): freq=0.48333333333333334 xi=0.36666666666666664 cap=0.008333333333333333
- Saka (arsenal): freq=0.4583333333333333 xi=0.44166666666666665 cap=0.058333333333333334
- A.Becker (liverpool): freq=0.44166666666666665 xi=0.4083333333333333 cap=0.0
- Haaland (man city): freq=0.425 xi=0.38333333333333336 cap=0.008333333333333333
- Barnes (newcastle): freq=0.4083333333333333 xi=0.39166666666666666 cap=0.016666666666666666
- João Pedro (chelsea): freq=0.4 xi=0.3416666666666667 cap=0.03333333333333333

## Mean-layer strands

- Squads: 120 · strands_with_squads=20
- Mean transfers / hits: 2.39 / 1.64
- Sources: ['cascade_long_opta', 'cascade_short_priced']
- Scoring: GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; mean score_team_side (score_xp_v2) on scenario λ moments; same-season as-of rates/priors/rolling m frozen at decision GW
- Top captain: {'name': 'Lewis-Potter', 'freq_captain': 0.4083333333333333}
- Elapsed: 466.1s

Top freq:

- Tzolakis (hull): freq=0.9333333333333333 xi=0.175 cap=0.0
- Lewis-Potter (brentford): freq=0.8583333333333333 xi=0.85 cap=0.4083333333333333
- Tarkowski (everton): freq=0.725 xi=0.725 cap=0.075
- Hall (newcastle): freq=0.7166666666666667 xi=0.6916666666666667 cap=0.008333333333333333
- Groß (brighton): freq=0.7 xi=0.7 cap=0.13333333333333333
- João Pedro (chelsea): freq=0.6666666666666666 xi=0.6666666666666666 cap=0.09166666666666666
- Haaland (man city): freq=0.625 xi=0.6166666666666667 cap=0.05
- Hume (sunderland): freq=0.6 xi=0.0 cap=0.0
- Trafford (leeds): freq=0.5583333333333333 xi=0.4583333333333333 cap=0.0
- Tanaka (leeds): freq=0.48333333333333334 xi=0.0 cap=0.0
- Dasilva (coventry): freq=0.4583333333333333 xi=0.20833333333333334 cap=0.025
- Havertz (arsenal): freq=0.44166666666666665 xi=0.44166666666666665 cap=0.041666666666666664
- Vuskovic (brighton): freq=0.4 xi=0.375 cap=0.0
- B.Fernandes (man utd): freq=0.375 xi=0.375 cap=0.10833333333333334
- Lukić (ipswich): freq=0.35833333333333334 xi=0.0 cap=0.0

Artifacts:
- `/workspace/data/predictions/2026-27/cascade_pathv_gen`
- `/workspace/data/predictions/2026-27/cascade_pathv_mean`

Reproduce:
```bash
PYTHONPATH=. python3 -m src.eval.cascade_pathv_gen_vs_mean
```
