# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **12** coherent strands: each fixture samples λ from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), Poisson scoreline, path rate update.
- Player xP: **`score_xp_v2`** via `numagent_xp.score_team_side` on Dixon–Coles moments from scenario λ; same-season as-of rates frozen at decision GW. Share seeding: **False**.
- Minutes overlay: **True** ({'n': 180, 'overlay': True, 'n_scaled': 19, 'n_zeroed': 4, 'mean_factor': 0.9569444444444445, 'status_counts': {'a': 161, 'd': 15, 'i': 3, 's': 1}}).
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`xi_plus_captain`).
- Solved squads: **72** · selectable **180** (59 with nonzero rG) · seed 7.

Scoring: `Opta match_prediction λ bands → Dixon–Coles M → numagent_xp.score_team_side (score_xp_v2); same-season as-of rates/priors/rolling m frozen at decision GW; FPL availability overlay on m`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 572 | Tzolakis | hull | GKP | 4.7 | 0.79 | 0.11 | 0.00 | 57 |
| 2 | 229 | Tarkowski | everton | DEF | 6.2 | 0.78 | 0.78 | 0.15 | 56 |
| 3 | 124 | Groß | brighton | MID | 5.9 | 0.76 | 0.76 | 0.15 | 55 |
| 4 | 411 | Haaland | man city | FWD | 15.6 | 0.76 | 0.76 | 0.24 | 55 |
| 5 | 449 | Hall | newcastle | DEF | 5.3 | 0.72 | 0.71 | 0.03 | 52 |
| 6 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.61 | 0.61 | 0.18 | 44 |
| 7 | 534 | Hume | sunderland | DEF | 4.3 | 0.61 | 0.00 | 0.00 | 44 |
| 8 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.50 | 0.50 | 0.07 | 36 |
| 9 | 279 | Ajayi | hull | DEF | 4.2 | 0.47 | 0.00 | 0.00 | 34 |
| 10 | 385 | Trafford | leeds | GKP | 5.0 | 0.42 | 0.38 | 0.00 | 30 |
| 11 | 40 | Rogers | chelsea | MID | 7.8 | 0.40 | 0.40 | 0.10 | 29 |
| 12 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.40 | 0.00 | 0.00 | 29 |
| 13 | 504 | Vuskovic | brighton | DEF | 5.0 | 0.40 | 0.40 | 0.00 | 29 |
| 14 | 138 | Kostoulas | brighton | FWD | 5.6 | 0.39 | 0.39 | 0.00 | 28 |
| 15 | 12 | Saka | arsenal | MID | 9.6 | 0.35 | 0.35 | 0.00 | 25 |
| 16 | 200 | Lacroix | chelsea | DEF | 5.9 | 0.33 | 0.33 | 0.00 | 24 |
| 17 | 428 | Cunha | man utd | MID | 7.9 | 0.33 | 0.33 | 0.00 | 24 |
| 18 | 345 | Tanaka | leeds | MID | 4.9 | 0.32 | 0.00 | 0.00 | 23 |
| 19 | 316 | Emersonn | ipswich | FWD | 5.5 | 0.29 | 0.24 | 0.01 | 21 |
| 20 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.28 | 0.28 | 0.01 | 20 |
| 21 | 194 | Thomas-Asante | coventry | FWD | 5.0 | 0.26 | 0.00 | 0.00 | 19 |
| 22 | 176 | Dasilva | coventry | DEF | 4.0 | 0.26 | 0.21 | 0.00 | 19 |
| 23 | 464 | Wissa | newcastle | FWD | 6.2 | 0.25 | 0.24 | 0.03 | 18 |
| 24 | 154 | Palmer | chelsea | MID | 9.7 | 0.24 | 0.24 | 0.00 | 17 |
| 25 | 427 | Mbeumo | man utd | MID | 7.9 | 0.24 | 0.24 | 0.00 | 17 |
| 26 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.21 | 0.18 | 0.00 | 15 |
| 27 | 173 | Thomas | coventry | DEF | 4.0 | 0.21 | 0.14 | 0.00 | 15 |
| 28 | 286 | Belloumi | hull | MID | 5.1 | 0.21 | 0.21 | 0.03 | 15 |
| 29 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.19 | 0.19 | 0.00 | 14 |
| 30 | 335 | Stach | leeds | MID | 6.0 | 0.18 | 0.18 | 0.00 | 13 |
| 31 | 305 | Davis | ipswich | DEF | 4.0 | 0.17 | 0.08 | 0.00 | 12 |
| 32 | 250 | Leno | fulham | GKP | 4.5 | 0.15 | 0.00 | 0.00 | 11 |
| 33 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.12 | 0.01 | 0.00 | 9 |
| 34 | 453 | Barnes | newcastle | MID | 6.1 | 0.12 | 0.12 | 0.00 | 9 |
| 35 | 230 | Branthwaite | everton | DEF | 5.5 | 0.11 | 0.11 | 0.00 | 8 |
| 36 | 388 | Guéhi | man city | DEF | 6.0 | 0.11 | 0.11 | 0.00 | 8 |
| 37 | 267 | Lukić | ipswich | MID | 4.9 | 0.11 | 0.00 | 0.00 | 8 |
| 38 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.10 | 0.10 | 0.00 | 7 |
| 39 | 79 | Evanilson | bournemouth | FWD | 6.0 | 0.08 | 0.08 | 0.00 | 6 |
| 40 | 91 | Pinnock | coventry | DEF | 4.5 | 0.08 | 0.08 | 0.00 | 6 |
| … | (49 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain
```

