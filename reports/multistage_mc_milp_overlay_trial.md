# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`squad_sum`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **12** coherent strands: each fixture samples λ from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), Poisson scoreline, path rate update.
- Player xP: **`score_xp_v2`** via `numagent_xp.score_team_side` on Dixon–Coles moments from scenario λ; same-season as-of rates frozen at decision GW. Share seeding: **False**.
- Minutes overlay: **True** ({'n': 180, 'overlay': True, 'n_scaled': 19, 'n_zeroed': 4, 'mean_factor': 0.9569444444444445, 'status_counts': {'a': 161, 'd': 15, 'i': 3, 's': 1}}).
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`squad_sum`).
- Solved squads: **72** · selectable **180** (59 with nonzero rG) · seed 7.

Scoring: `Opta match_prediction λ bands → Dixon–Coles M → numagent_xp.score_team_side (score_xp_v2); same-season as-of rates/priors/rolling m frozen at decision GW; FPL availability overlay on m`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 229 | Tarkowski | everton | DEF | 6.2 | 0.83 | 0.00 | 0.00 | 60 |
| 2 | 449 | Hall | newcastle | DEF | 5.3 | 0.81 | 0.00 | 0.00 | 58 |
| 3 | 124 | Groß | brighton | MID | 5.9 | 0.81 | 0.00 | 0.00 | 58 |
| 4 | 411 | Haaland | man city | FWD | 15.6 | 0.72 | 0.00 | 0.00 | 52 |
| 5 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.68 | 0.00 | 0.00 | 49 |
| 6 | 385 | Trafford | leeds | GKP | 5.0 | 0.60 | 0.00 | 0.00 | 43 |
| 7 | 504 | Vuskovic | brighton | DEF | 5.0 | 0.54 | 0.00 | 0.00 | 39 |
| 8 | 138 | Kostoulas | brighton | FWD | 5.6 | 0.53 | 0.00 | 0.00 | 38 |
| 9 | 200 | Lacroix | chelsea | DEF | 5.9 | 0.47 | 0.00 | 0.00 | 34 |
| 10 | 40 | Rogers | chelsea | MID | 7.8 | 0.46 | 0.00 | 0.00 | 33 |
| 11 | 316 | Emersonn | ipswich | FWD | 5.5 | 0.46 | 0.00 | 0.00 | 33 |
| 12 | 428 | Cunha | man utd | MID | 7.9 | 0.39 | 0.00 | 0.00 | 28 |
| 13 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.35 | 0.00 | 0.00 | 25 |
| 14 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.33 | 0.00 | 0.00 | 24 |
| 15 | 464 | Wissa | newcastle | FWD | 6.2 | 0.33 | 0.00 | 0.00 | 24 |
| 16 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.32 | 0.00 | 0.00 | 23 |
| 17 | 427 | Mbeumo | man utd | MID | 7.9 | 0.29 | 0.00 | 0.00 | 21 |
| 18 | 572 | Tzolakis | hull | GKP | 4.7 | 0.29 | 0.00 | 0.00 | 21 |
| 19 | 12 | Saka | arsenal | MID | 9.6 | 0.29 | 0.00 | 0.00 | 21 |
| 20 | 176 | Dasilva | coventry | DEF | 4.0 | 0.28 | 0.00 | 0.00 | 20 |
| 21 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.28 | 0.00 | 0.00 | 20 |
| 22 | 286 | Belloumi | hull | MID | 5.1 | 0.25 | 0.00 | 0.00 | 18 |
| 23 | 388 | Guéhi | man city | DEF | 6.0 | 0.22 | 0.00 | 0.00 | 16 |
| 24 | 597 | Suzuki | aston villa | GKP | 5.0 | 0.21 | 0.00 | 0.00 | 15 |
| 25 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.21 | 0.00 | 0.00 | 15 |
| 26 | 389 | Matheus N. | man city | DEF | 5.9 | 0.21 | 0.00 | 0.00 | 15 |
| 27 | 173 | Thomas | coventry | DEF | 4.0 | 0.21 | 0.00 | 0.00 | 15 |
| 28 | 335 | Stach | leeds | MID | 6.0 | 0.21 | 0.00 | 0.00 | 15 |
| 29 | 154 | Palmer | chelsea | MID | 9.7 | 0.19 | 0.00 | 0.00 | 14 |
| 30 | 453 | Barnes | newcastle | MID | 6.1 | 0.19 | 0.00 | 0.00 | 14 |
| 31 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.18 | 0.00 | 0.00 | 13 |
| 32 | 79 | Evanilson | bournemouth | FWD | 6.0 | 0.15 | 0.00 | 0.00 | 11 |
| 33 | 334 | Muharemović | leeds | DEF | 5.0 | 0.15 | 0.00 | 0.00 | 11 |
| 34 | 230 | Branthwaite | everton | DEF | 5.5 | 0.15 | 0.00 | 0.00 | 11 |
| 35 | 106 | Thiago | brentford | FWD | 7.8 | 0.15 | 0.00 | 0.00 | 11 |
| 36 | 87 | Ajer | brentford | DEF | 4.5 | 0.14 | 0.00 | 0.00 | 10 |
| 37 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.14 | 0.00 | 0.00 | 10 |
| 38 | 567 | Horníček | newcastle | GKP | 5.0 | 0.12 | 0.00 | 0.00 | 9 |
| 39 | 332 | Justin | leeds | DEF | 4.5 | 0.11 | 0.00 | 0.00 | 8 |
| 40 | 82 | Kelleher | brentford | GKP | 5.0 | 0.11 | 0.00 | 0.00 | 8 |
| … | (34 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective squad_sum
```

