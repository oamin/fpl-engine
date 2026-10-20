# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain_ft`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **12** coherent strands: each fixture samples λ from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), Poisson scoreline, path rate update.
- Player xP: **`score_xp_v2`** via `numagent_xp.score_team_side` on Dixon–Coles moments from scenario λ; same-season as-of rates frozen at decision GW. Share seeding: **False**.
- Minutes overlay: **True** ({'n': 180, 'overlay': True, 'n_scaled': 19, 'n_zeroed': 4, 'mean_factor': 0.9569444444444445, 'status_counts': {'a': 161, 'd': 15, 'i': 3, 's': 1}}).
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`xi_plus_captain_ft`).
- Solved squads: **72** · selectable **180** (59 with nonzero rG) · seed 7.

Scoring: `Opta match_prediction λ bands → Dixon–Coles M → numagent_xp.score_team_side (score_xp_v2); same-season as-of rates/priors/rolling m frozen at decision GW; FPL availability overlay on m`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 229 | Tarkowski | everton | DEF | 6.2 | 0.79 | 0.79 | 0.15 | 57 |
| 2 | 124 | Groß | brighton | MID | 5.9 | 0.76 | 0.76 | 0.15 | 55 |
| 3 | 411 | Haaland | man city | FWD | 15.6 | 0.71 | 0.71 | 0.24 | 51 |
| 4 | 449 | Hall | newcastle | DEF | 5.3 | 0.68 | 0.68 | 0.03 | 49 |
| 5 | 572 | Tzolakis | hull | GKP | 4.7 | 0.68 | 0.12 | 0.00 | 49 |
| 6 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.61 | 0.61 | 0.18 | 44 |
| 7 | 316 | Emersonn | ipswich | FWD | 5.5 | 0.53 | 0.33 | 0.01 | 38 |
| 8 | 385 | Trafford | leeds | GKP | 5.0 | 0.53 | 0.44 | 0.00 | 38 |
| 9 | 40 | Rogers | chelsea | MID | 7.8 | 0.51 | 0.44 | 0.10 | 37 |
| 10 | 200 | Lacroix | chelsea | DEF | 5.9 | 0.50 | 0.43 | 0.00 | 36 |
| 11 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.50 | 0.50 | 0.07 | 36 |
| 12 | 176 | Dasilva | coventry | DEF | 4.0 | 0.49 | 0.25 | 0.00 | 35 |
| 13 | 534 | Hume | sunderland | DEF | 4.3 | 0.47 | 0.00 | 0.00 | 34 |
| 14 | 428 | Cunha | man utd | MID | 7.9 | 0.46 | 0.42 | 0.00 | 33 |
| 15 | 138 | Kostoulas | brighton | FWD | 5.6 | 0.38 | 0.35 | 0.00 | 27 |
| 16 | 305 | Davis | ipswich | DEF | 4.0 | 0.35 | 0.08 | 0.00 | 25 |
| 17 | 173 | Thomas | coventry | DEF | 4.0 | 0.33 | 0.21 | 0.00 | 24 |
| 18 | 12 | Saka | arsenal | MID | 9.6 | 0.31 | 0.31 | 0.00 | 22 |
| 19 | 464 | Wissa | newcastle | FWD | 6.2 | 0.31 | 0.26 | 0.03 | 22 |
| 20 | 504 | Vuskovic | brighton | DEF | 5.0 | 0.31 | 0.31 | 0.00 | 22 |
| 21 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.29 | 0.29 | 0.01 | 21 |
| 22 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.26 | 0.00 | 0.00 | 19 |
| 23 | 427 | Mbeumo | man utd | MID | 7.9 | 0.26 | 0.26 | 0.00 | 19 |
| 24 | 154 | Palmer | chelsea | MID | 9.7 | 0.24 | 0.22 | 0.00 | 17 |
| 25 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.24 | 0.12 | 0.00 | 17 |
| 26 | 286 | Belloumi | hull | MID | 5.1 | 0.24 | 0.21 | 0.03 | 17 |
| 27 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.19 | 0.19 | 0.00 | 14 |
| 28 | 453 | Barnes | newcastle | MID | 6.1 | 0.18 | 0.15 | 0.00 | 13 |
| 29 | 79 | Evanilson | bournemouth | FWD | 6.0 | 0.17 | 0.11 | 0.00 | 12 |
| 30 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.17 | 0.14 | 0.00 | 12 |
| 31 | 91 | Pinnock | coventry | DEF | 4.5 | 0.15 | 0.08 | 0.00 | 11 |
| 32 | 250 | Leno | fulham | GKP | 4.5 | 0.15 | 0.00 | 0.00 | 11 |
| 33 | 388 | Guéhi | man city | DEF | 6.0 | 0.14 | 0.12 | 0.00 | 10 |
| 34 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.12 | 0.00 | 0.00 | 9 |
| 35 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.12 | 0.10 | 0.00 | 9 |
| 36 | 267 | Lukić | ipswich | MID | 4.9 | 0.12 | 0.00 | 0.00 | 9 |
| 37 | 567 | Horníček | newcastle | GKP | 5.0 | 0.12 | 0.11 | 0.00 | 9 |
| 38 | 230 | Branthwaite | everton | DEF | 5.5 | 0.11 | 0.08 | 0.00 | 8 |
| 39 | 249 | Barry | everton | FWD | 5.7 | 0.11 | 0.06 | 0.00 | 8 |
| 40 | 295 | McBurnie | hull | FWD | 5.5 | 0.10 | 0.00 | 0.00 | 7 |
| … | (28 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain_ft
```

