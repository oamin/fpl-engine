# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain_ft`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **12** coherent strands: priced weeks use **cascade_short** (`priced`={'gws': [8], 'n_fits': 10, 'n_props_matched': 330, 'odds_path': 'data/scratch/gw08_odds_full.json'}); unpriced weeks use **cascade_long** Opta place→GF/GA±MAE → `M` → `k ∼ M`.
- Player xP: **GW8 cascade_short priced λ±15%→M→k (props matched=330); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→attack→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW** (props=True; generative=True). Share seeding: **False**.
- Minutes overlay: **False** ({'overlay': False, 'n': 180}).
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`xi_plus_captain_ft`).
- Solved squads: **72** · selectable **180** (59 with nonzero rG) · seed 7.

Scoring: `GW8 cascade_short priced λ±15%→M→k (props matched=330); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→attack→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 229 | Tarkowski | everton | DEF | 6.2 | 0.90 | 0.85 | 0.10 | 65 |
| 2 | 449 | Hall | newcastle | DEF | 5.3 | 0.88 | 0.79 | 0.08 | 63 |
| 3 | 124 | Groß | brighton | MID | 5.9 | 0.85 | 0.82 | 0.19 | 61 |
| 4 | 112 | Van Hecke | spurs | DEF | 4.9 | 0.79 | 0.67 | 0.11 | 57 |
| 5 | 176 | Dasilva | coventry | DEF | 4.0 | 0.79 | 0.65 | 0.11 | 57 |
| 6 | 572 | Tzolakis | hull | GKP | 4.7 | 0.74 | 0.04 | 0.00 | 53 |
| 7 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.72 | 0.64 | 0.00 | 52 |
| 8 | 286 | Belloumi | hull | MID | 5.1 | 0.65 | 0.50 | 0.07 | 47 |
| 9 | 12 | Saka | arsenal | MID | 9.6 | 0.61 | 0.60 | 0.04 | 44 |
| 10 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.56 | 0.01 | 0.00 | 40 |
| 11 | 366 | Wirtz | liverpool | MID | 7.3 | 0.53 | 0.51 | 0.07 | 38 |
| 12 | 534 | Hume | sunderland | DEF | 4.3 | 0.53 | 0.00 | 0.00 | 38 |
| 13 | 453 | Barnes | newcastle | MID | 6.1 | 0.46 | 0.43 | 0.06 | 33 |
| 14 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.43 | 0.39 | 0.01 | 31 |
| 15 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.39 | 0.04 | 0.00 | 28 |
| 16 | 411 | Haaland | man city | FWD | 15.6 | 0.39 | 0.36 | 0.01 | 28 |
| 17 | 305 | Davis | ipswich | DEF | 4.0 | 0.36 | 0.25 | 0.00 | 26 |
| 18 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.36 | 0.36 | 0.06 | 26 |
| 19 | 138 | Kostoulas | brighton | FWD | 5.6 | 0.33 | 0.17 | 0.00 | 24 |
| 20 | 110 | Rushworth | coventry | GKP | 4.5 | 0.29 | 0.21 | 0.00 | 21 |
| 21 | 40 | Rogers | chelsea | MID | 7.8 | 0.28 | 0.26 | 0.00 | 20 |
| 22 | 154 | Palmer | chelsea | MID | 9.7 | 0.21 | 0.21 | 0.01 | 15 |
| 23 | 552 | Brobbey | sunderland | FWD | 5.7 | 0.21 | 0.14 | 0.00 | 15 |
| 24 | 166 | N.Jackson | aston villa | FWD | 6.5 | 0.19 | 0.18 | 0.01 | 14 |
| 25 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.18 | 0.18 | 0.00 | 13 |
| 26 | 249 | Barry | everton | FWD | 5.7 | 0.15 | 0.06 | 0.00 | 11 |
| 27 | 397 | Semenyo | man city | MID | 8.4 | 0.15 | 0.14 | 0.00 | 11 |
| 28 | 399 | Cherki | man city | MID | 7.8 | 0.12 | 0.12 | 0.00 | 9 |
| 29 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.12 | 0.11 | 0.01 | 9 |
| 30 | 385 | Trafford | leeds | GKP | 5.0 | 0.12 | 0.03 | 0.00 | 9 |
| 31 | 204 | Mitchell | crystal palace | DEF | 4.5 | 0.12 | 0.10 | 0.01 | 9 |
| 32 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.11 | 0.11 | 0.00 | 8 |
| 33 | 279 | Ajayi | hull | DEF | 4.2 | 0.11 | 0.07 | 0.00 | 8 |
| 34 | 316 | Emersonn | ipswich | FWD | 5.5 | 0.10 | 0.07 | 0.00 | 7 |
| 35 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.10 | 0.10 | 0.01 | 7 |
| 36 | 588 | Khalaili | crystal palace | DEF | 5.0 | 0.08 | 0.06 | 0.00 | 6 |
| 37 | 356 | Virgil | liverpool | DEF | 6.5 | 0.08 | 0.08 | 0.00 | 6 |
| 38 | 194 | Thomas-Asante | coventry | FWD | 5.0 | 0.08 | 0.00 | 0.00 | 6 |
| 39 | 345 | Tanaka | leeds | MID | 4.9 | 0.07 | 0.00 | 0.00 | 5 |
| 40 | 268 | King | fulham | MID | 5.6 | 0.07 | 0.07 | 0.00 | 5 |
| … | (24 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain_ft
```

