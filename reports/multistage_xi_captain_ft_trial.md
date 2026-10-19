# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain_ft`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **6** coherent strands: each fixture samples λ from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), Poisson scoreline, path rate update.
- Player xP: frozen deadline shares/xmi/defcon (`add_player_priors`) × scenario pot via `xp_on_pot` (score_xp v2). Share seeding: **False**.
- Minutes overlay: **False** ({'overlay': False, 'n': 180}).
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`xi_plus_captain_ft`).
- Solved squads: **36** · selectable **180** (170 with nonzero share) · seed 7.

Scoring: `Opta match_prediction λ bands → sampled scenario pot; deadline add_player_priors shares/xmi/defcon × xp_on_pot (score_xp v2)`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 572 | Tzolakis | hull | GKP | 4.7 | 1.00 | 0.08 | 0.00 | 36 |
| 2 | 40 | Rogers | chelsea | MID | 7.8 | 0.72 | 0.69 | 0.03 | 26 |
| 3 | 449 | Hall | newcastle | DEF | 5.3 | 0.72 | 0.69 | 0.00 | 26 |
| 4 | 249 | Barry | everton | FWD | 5.7 | 0.69 | 0.64 | 0.00 | 25 |
| 5 | 12 | Saka | arsenal | MID | 9.6 | 0.67 | 0.67 | 0.50 | 24 |
| 6 | 427 | Mbeumo | man utd | MID | 7.9 | 0.64 | 0.64 | 0.14 | 23 |
| 7 | 411 | Haaland | man city | FWD | 15.6 | 0.64 | 0.61 | 0.14 | 23 |
| 8 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.58 | 0.22 | 0.00 | 21 |
| 9 | 566 | Silva | bournemouth | DEF | 5.0 | 0.53 | 0.53 | 0.08 | 19 |
| 10 | 464 | Wissa | newcastle | FWD | 6.2 | 0.53 | 0.47 | 0.03 | 19 |
| 11 | 534 | Hume | sunderland | DEF | 4.3 | 0.50 | 0.00 | 0.00 | 18 |
| 12 | 305 | Davis | ipswich | DEF | 4.0 | 0.47 | 0.39 | 0.06 | 17 |
| 13 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.47 | 0.44 | 0.00 | 17 |
| 14 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.36 | 0.33 | 0.03 | 13 |
| 15 | 200 | Lacroix | chelsea | DEF | 5.9 | 0.36 | 0.33 | 0.00 | 13 |
| 16 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.33 | 0.33 | 0.00 | 12 |
| 17 | 542 | E.Le Fée | sunderland | MID | 5.7 | 0.31 | 0.25 | 0.00 | 11 |
| 18 | 588 | Khalaili | crystal palace | DEF | 5.0 | 0.28 | 0.19 | 0.00 | 10 |
| 19 | 183 | Rudoni | coventry | MID | 4.9 | 0.28 | 0.06 | 0.00 | 10 |
| 20 | 567 | Horníček | newcastle | GKP | 5.0 | 0.28 | 0.28 | 0.00 | 10 |
| 21 | 286 | Belloumi | hull | MID | 5.1 | 0.28 | 0.25 | 0.00 | 10 |
| 22 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.25 | 0.03 | 0.00 | 9 |
| 23 | 41 | Buendía | aston villa | MID | 5.9 | 0.25 | 0.19 | 0.00 | 9 |
| 24 | 345 | Tanaka | leeds | MID | 4.9 | 0.22 | 0.00 | 0.00 | 8 |
| 25 | 591 | Enciso | ipswich | MID | 5.5 | 0.19 | 0.08 | 0.00 | 7 |
| 26 | 60 | Hill | bournemouth | DEF | 5.5 | 0.19 | 0.11 | 0.00 | 7 |
| 27 | 173 | Thomas | coventry | DEF | 4.0 | 0.19 | 0.11 | 0.00 | 7 |
| 28 | 109 | Verbruggen | brighton | GKP | 4.5 | 0.19 | 0.19 | 0.00 | 7 |
| 29 | 335 | Stach | leeds | MID | 6.0 | 0.19 | 0.17 | 0.00 | 7 |
| 30 | 154 | Palmer | chelsea | MID | 9.7 | 0.17 | 0.17 | 0.00 | 6 |
| 31 | 259 | Diop | ipswich | DEF | 4.0 | 0.17 | 0.03 | 0.00 | 6 |
| 32 | 4 | Gabriel | arsenal | DEF | 8.0 | 0.17 | 0.14 | 0.00 | 6 |
| 33 | 124 | Groß | brighton | MID | 5.9 | 0.17 | 0.14 | 0.00 | 6 |
| 34 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.17 | 0.17 | 0.00 | 6 |
| 35 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.17 | 0.14 | 0.00 | 6 |
| 36 | 279 | Ajayi | hull | DEF | 4.2 | 0.17 | 0.06 | 0.00 | 6 |
| 37 | 552 | Brobbey | sunderland | FWD | 5.7 | 0.17 | 0.08 | 0.00 | 6 |
| 38 | 388 | Guéhi | man city | DEF | 6.0 | 0.14 | 0.11 | 0.00 | 5 |
| 39 | 1 | Raya | arsenal | GKP | 6.1 | 0.14 | 0.14 | 0.00 | 5 |
| 40 | 330 | Bogle | leeds | DEF | 4.6 | 0.14 | 0.14 | 0.00 | 5 |
| … | (14 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain_ft
```

