# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain_ft`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **12** coherent strands: priced weeks use **cascade_short** (`priced`={'gws': [8], 'n_fits': 10, 'n_props_matched': 330, 'odds_path': '/workspace/data/scratch/gw08_odds_full.json'}); unpriced weeks use **cascade_long** Opta place→GF/GA±MAE → `M` → `k ∼ M`.
- Player xP: **GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→attack→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW** (props=True; generative=True). Share seeding: **False**.
- Minutes overlay: **False** ({'overlay': False, 'n': 180}).
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`xi_plus_captain_ft`).
- Solved squads: **72** · selectable **180** (59 with nonzero rG) · seed 7.

Scoring: `GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→attack→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 124 | Groß | brighton | MID | 5.9 | 0.93 | 0.93 | 0.14 | 67 |
| 2 | 572 | Tzolakis | hull | GKP | 4.7 | 0.92 | 0.14 | 0.00 | 66 |
| 3 | 229 | Tarkowski | everton | DEF | 6.2 | 0.89 | 0.86 | 0.21 | 64 |
| 4 | 176 | Dasilva | coventry | DEF | 4.0 | 0.85 | 0.68 | 0.12 | 61 |
| 5 | 449 | Hall | newcastle | DEF | 5.3 | 0.85 | 0.75 | 0.11 | 61 |
| 6 | 112 | Van Hecke | spurs | DEF | 4.9 | 0.75 | 0.64 | 0.11 | 54 |
| 7 | 366 | Wirtz | liverpool | MID | 7.3 | 0.74 | 0.68 | 0.06 | 53 |
| 8 | 12 | Saka | arsenal | MID | 9.6 | 0.56 | 0.53 | 0.07 | 40 |
| 9 | 534 | Hume | sunderland | DEF | 4.3 | 0.56 | 0.00 | 0.00 | 40 |
| 10 | 166 | N.Jackson | aston villa | FWD | 6.5 | 0.51 | 0.40 | 0.01 | 37 |
| 11 | 286 | Belloumi | hull | MID | 5.1 | 0.51 | 0.44 | 0.00 | 37 |
| 12 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.42 | 0.39 | 0.00 | 30 |
| 13 | 411 | Haaland | man city | FWD | 15.6 | 0.40 | 0.38 | 0.01 | 29 |
| 14 | 110 | Rushworth | coventry | GKP | 4.5 | 0.39 | 0.28 | 0.00 | 28 |
| 15 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.39 | 0.11 | 0.00 | 28 |
| 16 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.38 | 0.00 | 0.00 | 27 |
| 17 | 138 | Kostoulas | brighton | FWD | 5.6 | 0.35 | 0.21 | 0.01 | 25 |
| 18 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.35 | 0.32 | 0.03 | 25 |
| 19 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.31 | 0.28 | 0.01 | 22 |
| 20 | 305 | Davis | ipswich | DEF | 4.0 | 0.28 | 0.21 | 0.01 | 20 |
| 21 | 453 | Barnes | newcastle | MID | 6.1 | 0.25 | 0.24 | 0.00 | 18 |
| 22 | 40 | Rogers | chelsea | MID | 7.8 | 0.21 | 0.21 | 0.00 | 15 |
| 23 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.21 | 0.18 | 0.00 | 15 |
| 24 | 345 | Tanaka | leeds | MID | 4.9 | 0.19 | 0.00 | 0.00 | 14 |
| 25 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.19 | 0.17 | 0.00 | 14 |
| 26 | 399 | Cherki | man city | MID | 7.8 | 0.18 | 0.18 | 0.00 | 13 |
| 27 | 249 | Barry | everton | FWD | 5.7 | 0.18 | 0.07 | 0.00 | 13 |
| 28 | 397 | Semenyo | man city | MID | 8.4 | 0.15 | 0.14 | 0.00 | 11 |
| 29 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.15 | 0.14 | 0.00 | 11 |
| 30 | 279 | Ajayi | hull | DEF | 4.2 | 0.14 | 0.08 | 0.00 | 10 |
| 31 | 362 | Jacquet | liverpool | DEF | 5.0 | 0.14 | 0.12 | 0.01 | 10 |
| 32 | 154 | Palmer | chelsea | MID | 9.7 | 0.14 | 0.12 | 0.00 | 10 |
| 33 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.11 | 0.10 | 0.03 | 8 |
| 34 | 106 | Thiago | brentford | FWD | 7.8 | 0.11 | 0.10 | 0.00 | 8 |
| 35 | 79 | Evanilson | bournemouth | FWD | 6.0 | 0.10 | 0.08 | 0.00 | 7 |
| 36 | 267 | Lukić | ipswich | MID | 4.9 | 0.10 | 0.00 | 0.00 | 7 |
| 37 | 597 | Suzuki | aston villa | GKP | 5.0 | 0.08 | 0.04 | 0.00 | 6 |
| 38 | 385 | Trafford | leeds | GKP | 5.0 | 0.08 | 0.08 | 0.00 | 6 |
| 39 | 388 | Guéhi | man city | DEF | 6.0 | 0.08 | 0.07 | 0.00 | 6 |
| 40 | 204 | Mitchell | crystal palace | DEF | 4.5 | 0.08 | 0.06 | 0.00 | 6 |
| … | (27 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain_ft
```

