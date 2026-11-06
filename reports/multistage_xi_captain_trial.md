# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **12** coherent strands: priced weeks use **cascade_short** (`priced`={'gws': [8], 'n_fits': 10, 'n_props_matched': 330, 'odds_path': '/workspace/data/scratch/gw08_odds_full.json'}); unpriced weeks use **cascade_long** Opta place→GF/GA±MAE → `M` → `k ∼ M`.
- Player xP: **GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→attack→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW** (props=True; generative=True). Share seeding: **False**.
- Minutes overlay: **False** ({'overlay': False, 'n': 180}).
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`xi_plus_captain`).
- Solved squads: **72** · selectable **180** (59 with nonzero rG) · seed 7.

Scoring: `GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→attack→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 572 | Tzolakis | hull | GKP | 4.7 | 0.90 | 0.00 | 0.00 | 65 |
| 2 | 124 | Groß | brighton | MID | 5.9 | 0.81 | 0.81 | 0.14 | 58 |
| 3 | 229 | Tarkowski | everton | DEF | 6.2 | 0.76 | 0.76 | 0.21 | 55 |
| 4 | 534 | Hume | sunderland | DEF | 4.3 | 0.69 | 0.00 | 0.00 | 50 |
| 5 | 449 | Hall | newcastle | DEF | 5.3 | 0.68 | 0.68 | 0.11 | 49 |
| 6 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.68 | 0.00 | 0.00 | 49 |
| 7 | 366 | Wirtz | liverpool | MID | 7.3 | 0.60 | 0.60 | 0.06 | 43 |
| 8 | 112 | Van Hecke | spurs | DEF | 4.9 | 0.58 | 0.58 | 0.11 | 42 |
| 9 | 176 | Dasilva | coventry | DEF | 4.0 | 0.57 | 0.56 | 0.12 | 41 |
| 10 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.46 | 0.06 | 0.00 | 33 |
| 11 | 345 | Tanaka | leeds | MID | 4.9 | 0.44 | 0.00 | 0.00 | 32 |
| 12 | 12 | Saka | arsenal | MID | 9.6 | 0.42 | 0.42 | 0.07 | 30 |
| 13 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.39 | 0.39 | 0.00 | 28 |
| 14 | 286 | Belloumi | hull | MID | 5.1 | 0.39 | 0.39 | 0.00 | 28 |
| 15 | 166 | N.Jackson | aston villa | FWD | 6.5 | 0.36 | 0.36 | 0.01 | 26 |
| 16 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.35 | 0.35 | 0.01 | 25 |
| 17 | 411 | Haaland | man city | FWD | 15.6 | 0.33 | 0.33 | 0.01 | 24 |
| 18 | 453 | Barnes | newcastle | MID | 6.1 | 0.32 | 0.32 | 0.00 | 23 |
| 19 | 279 | Ajayi | hull | DEF | 4.2 | 0.31 | 0.08 | 0.00 | 22 |
| 20 | 110 | Rushworth | coventry | GKP | 4.5 | 0.29 | 0.29 | 0.00 | 21 |
| 21 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.28 | 0.28 | 0.03 | 20 |
| 22 | 305 | Davis | ipswich | DEF | 4.0 | 0.24 | 0.24 | 0.01 | 17 |
| 23 | 138 | Kostoulas | brighton | FWD | 5.6 | 0.21 | 0.21 | 0.01 | 15 |
| 24 | 40 | Rogers | chelsea | MID | 7.8 | 0.21 | 0.21 | 0.00 | 15 |
| 25 | 397 | Semenyo | man city | MID | 8.4 | 0.17 | 0.17 | 0.00 | 12 |
| 26 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.17 | 0.17 | 0.00 | 12 |
| 27 | 399 | Cherki | man city | MID | 7.8 | 0.17 | 0.17 | 0.00 | 12 |
| 28 | 154 | Palmer | chelsea | MID | 9.7 | 0.17 | 0.17 | 0.00 | 12 |
| 29 | 388 | Guéhi | man city | DEF | 6.0 | 0.15 | 0.15 | 0.00 | 11 |
| 30 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.15 | 0.15 | 0.00 | 11 |
| 31 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.15 | 0.15 | 0.00 | 11 |
| 32 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.12 | 0.12 | 0.03 | 9 |
| 33 | 249 | Barry | everton | FWD | 5.7 | 0.12 | 0.10 | 0.00 | 9 |
| 34 | 362 | Jacquet | liverpool | DEF | 5.0 | 0.12 | 0.12 | 0.01 | 9 |
| 35 | 385 | Trafford | leeds | GKP | 5.0 | 0.12 | 0.08 | 0.00 | 9 |
| 36 | 316 | Emersonn | ipswich | FWD | 5.5 | 0.11 | 0.08 | 0.00 | 8 |
| 37 | 356 | Virgil | liverpool | DEF | 6.5 | 0.11 | 0.11 | 0.01 | 8 |
| 38 | 597 | Suzuki | aston villa | GKP | 5.0 | 0.11 | 0.11 | 0.00 | 8 |
| 39 | 267 | Lukić | ipswich | MID | 4.9 | 0.10 | 0.00 | 0.00 | 7 |
| 40 | 106 | Thiago | brentford | FWD | 7.8 | 0.10 | 0.10 | 0.00 | 7 |
| … | (45 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain
```

