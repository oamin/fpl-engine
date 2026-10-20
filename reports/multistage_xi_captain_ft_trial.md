# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain_ft`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **12** coherent strands: each fixture samples λ from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), Poisson scoreline, path rate update.
- Player xP: **`score_xp_v2`** via `numagent_xp.score_team_side` on Dixon–Coles moments from scenario λ; same-season as-of rates frozen at decision GW. Share seeding: **False**.
- Minutes overlay: **False** ({'overlay': False, 'n': 180}).
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`xi_plus_captain_ft`).
- Solved squads: **72** · selectable **180** (59 with nonzero rG) · seed 7.

Scoring: `Opta match_prediction λ bands → Dixon–Coles M → numagent_xp.score_team_side (score_xp_v2); same-season as-of rates/priors/rolling m frozen at decision GW`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 229 | Tarkowski | everton | DEF | 6.2 | 0.79 | 0.78 | 0.17 | 57 |
| 2 | 124 | Groß | brighton | MID | 5.9 | 0.76 | 0.76 | 0.15 | 55 |
| 3 | 572 | Tzolakis | hull | GKP | 4.7 | 0.76 | 0.12 | 0.00 | 55 |
| 4 | 449 | Hall | newcastle | DEF | 5.3 | 0.71 | 0.71 | 0.01 | 51 |
| 5 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.68 | 0.68 | 0.18 | 49 |
| 6 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.67 | 0.67 | 0.14 | 48 |
| 7 | 411 | Haaland | man city | FWD | 15.6 | 0.61 | 0.61 | 0.19 | 44 |
| 8 | 428 | Cunha | man utd | MID | 7.9 | 0.56 | 0.44 | 0.00 | 40 |
| 9 | 385 | Trafford | leeds | GKP | 5.0 | 0.54 | 0.42 | 0.00 | 39 |
| 10 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.51 | 0.51 | 0.07 | 37 |
| 11 | 176 | Dasilva | coventry | DEF | 4.0 | 0.49 | 0.24 | 0.00 | 35 |
| 12 | 534 | Hume | sunderland | DEF | 4.3 | 0.44 | 0.00 | 0.00 | 32 |
| 13 | 173 | Thomas | coventry | DEF | 4.0 | 0.40 | 0.21 | 0.00 | 29 |
| 14 | 154 | Palmer | chelsea | MID | 9.7 | 0.39 | 0.36 | 0.00 | 28 |
| 15 | 504 | Vuskovic | brighton | DEF | 5.0 | 0.36 | 0.36 | 0.00 | 26 |
| 16 | 427 | Mbeumo | man utd | MID | 7.9 | 0.35 | 0.32 | 0.00 | 25 |
| 17 | 40 | Rogers | chelsea | MID | 7.8 | 0.35 | 0.21 | 0.00 | 25 |
| 18 | 200 | Lacroix | chelsea | DEF | 5.9 | 0.31 | 0.29 | 0.00 | 22 |
| 19 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.31 | 0.19 | 0.00 | 22 |
| 20 | 138 | Kostoulas | brighton | FWD | 5.6 | 0.31 | 0.29 | 0.00 | 22 |
| 21 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.29 | 0.00 | 0.00 | 21 |
| 22 | 286 | Belloumi | hull | MID | 5.1 | 0.29 | 0.21 | 0.04 | 21 |
| 23 | 26 | Havertz | arsenal | FWD | 7.6 | 0.25 | 0.25 | 0.00 | 18 |
| 24 | 305 | Davis | ipswich | DEF | 4.0 | 0.21 | 0.06 | 0.00 | 15 |
| 25 | 316 | Emersonn | ipswich | FWD | 5.5 | 0.21 | 0.12 | 0.00 | 15 |
| 26 | 279 | Ajayi | hull | DEF | 4.2 | 0.21 | 0.00 | 0.00 | 15 |
| 27 | 464 | Wissa | newcastle | FWD | 6.2 | 0.19 | 0.19 | 0.03 | 14 |
| 28 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.19 | 0.17 | 0.00 | 14 |
| 29 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.18 | 0.18 | 0.01 | 13 |
| 30 | 335 | Stach | leeds | MID | 6.0 | 0.17 | 0.14 | 0.00 | 12 |
| 31 | 453 | Barnes | newcastle | MID | 6.1 | 0.17 | 0.17 | 0.00 | 12 |
| 32 | 230 | Branthwaite | everton | DEF | 5.5 | 0.15 | 0.08 | 0.00 | 11 |
| 33 | 552 | Brobbey | sunderland | FWD | 5.7 | 0.14 | 0.11 | 0.00 | 10 |
| 34 | 267 | Lukić | ipswich | MID | 4.9 | 0.12 | 0.00 | 0.00 | 9 |
| 35 | 567 | Horníček | newcastle | GKP | 5.0 | 0.12 | 0.11 | 0.00 | 9 |
| 36 | 12 | Saka | arsenal | MID | 9.6 | 0.11 | 0.11 | 0.00 | 8 |
| 37 | 345 | Tanaka | leeds | MID | 4.9 | 0.10 | 0.00 | 0.00 | 7 |
| 38 | 362 | Jacquet | liverpool | DEF | 5.0 | 0.10 | 0.06 | 0.00 | 7 |
| 39 | 332 | Justin | leeds | DEF | 4.5 | 0.08 | 0.04 | 0.00 | 6 |
| 40 | 79 | Evanilson | bournemouth | FWD | 6.0 | 0.08 | 0.03 | 0.00 | 6 |
| … | (35 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain_ft
```

