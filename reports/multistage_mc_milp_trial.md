# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **20** coherent strands: each fixture samples λ from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), Poisson scoreline, path rate update.
- Player xP: frozen deadline shares/xmi/defcon (`add_player_priors`) × scenario pot via `xp_on_pot` (score_xp v2). Share seeding: **False**.
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP.
- Solved squads: **120** · selectable **180** (170 with nonzero share) · seed 7.

Scoring: `Opta match_prediction λ bands → sampled scenario pot; deadline add_player_priors shares/xmi/defcon × xp_on_pot (score_xp v2)`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | n |
|---:|---:|---|---|---|---:|---:|---:|
| 1 | 12 | Saka | arsenal | MID | 9.6 | 0.89 | 107 |
| 2 | 249 | Barry | everton | FWD | 5.7 | 0.81 | 97 |
| 3 | 40 | Rogers | chelsea | MID | 7.8 | 0.73 | 88 |
| 4 | 449 | Hall | newcastle | DEF | 5.3 | 0.66 | 79 |
| 5 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.65 | 78 |
| 6 | 566 | Silva | bournemouth | DEF | 5.0 | 0.63 | 76 |
| 7 | 464 | Wissa | newcastle | FWD | 6.2 | 0.63 | 76 |
| 8 | 427 | Mbeumo | man utd | MID | 7.9 | 0.61 | 73 |
| 9 | 411 | Haaland | man city | FWD | 15.6 | 0.56 | 67 |
| 10 | 109 | Verbruggen | brighton | GKP | 4.5 | 0.51 | 61 |
| 11 | 335 | Stach | leeds | MID | 6.0 | 0.44 | 53 |
| 12 | 305 | Davis | ipswich | DEF | 4.0 | 0.40 | 48 |
| 13 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.40 | 48 |
| 14 | 588 | Khalaili | crystal palace | DEF | 5.0 | 0.38 | 45 |
| 15 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.36 | 43 |
| 16 | 124 | Groß | brighton | MID | 5.9 | 0.35 | 42 |
| 17 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.33 | 40 |
| 18 | 200 | Lacroix | chelsea | DEF | 5.9 | 0.32 | 38 |
| 19 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.24 | 29 |
| 20 | 82 | Kelleher | brentford | GKP | 5.0 | 0.24 | 29 |
| 21 | 388 | Guéhi | man city | DEF | 6.0 | 0.23 | 28 |
| 22 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.22 | 26 |
| 23 | 330 | Bogle | leeds | DEF | 4.6 | 0.22 | 26 |
| 24 | 41 | Buendía | aston villa | MID | 5.9 | 0.19 | 23 |
| 25 | 384 | Donnarumma | man city | GKP | 5.5 | 0.19 | 23 |
| 26 | 567 | Horníček | newcastle | GKP | 5.0 | 0.18 | 22 |
| 27 | 183 | Rudoni | coventry | MID | 4.9 | 0.18 | 22 |
| 28 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.18 | 22 |
| 29 | 173 | Thomas | coventry | DEF | 4.0 | 0.17 | 21 |
| 30 | 356 | Virgil | liverpool | DEF | 6.5 | 0.17 | 20 |
| 31 | 286 | Belloumi | hull | MID | 5.1 | 0.16 | 19 |
| 32 | 106 | Thiago | brentford | FWD | 7.8 | 0.15 | 18 |
| 33 | 542 | E.Le Fée | sunderland | MID | 5.7 | 0.15 | 18 |
| 34 | 154 | Palmer | chelsea | MID | 9.7 | 0.14 | 17 |
| 35 | 4 | Gabriel | arsenal | DEF | 8.0 | 0.14 | 17 |
| 36 | 591 | Enciso | ipswich | MID | 5.5 | 0.13 | 16 |
| 37 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.13 | 16 |
| 38 | 412 | Lammens | man utd | GKP | 4.9 | 0.12 | 14 |
| 39 | 60 | Hill | bournemouth | DEF | 5.5 | 0.11 | 13 |
| 40 | 334 | Muharemović | leeds | DEF | 5.0 | 0.09 | 11 |
| … | (43 more) | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Low freq = path-sensitive / avoid locking capital.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20
```

