# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **8** coherent strands: each fixture samples λ from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), Poisson scoreline, path rate update.
- Player xP: frozen deadline shares/xmi/defcon (`add_player_priors`) × scenario pot via `xp_on_pot` (score_xp v2). Share seeding: **False**.
- Snapshots at GW [8, 11, 14, 17] → selectable-pool MILP.
- Solved squads: **32** · selectable **180** (170 with nonzero share) · seed 7.

Scoring: `Opta match_prediction λ bands → sampled scenario pot; deadline add_player_priors shares/xmi/defcon × xp_on_pot (score_xp v2)`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | n |
|---:|---:|---|---|---|---:|---:|---:|
| 1 | 249 | Barry | everton | FWD | 5.7 | 0.78 | 25 |
| 2 | 449 | Hall | newcastle | DEF | 5.3 | 0.75 | 24 |
| 3 | 12 | Saka | arsenal | MID | 9.6 | 0.75 | 24 |
| 4 | 566 | Silva | bournemouth | DEF | 5.0 | 0.72 | 23 |
| 5 | 40 | Rogers | chelsea | MID | 7.8 | 0.72 | 23 |
| 6 | 464 | Wissa | newcastle | FWD | 6.2 | 0.66 | 21 |
| 7 | 427 | Mbeumo | man utd | MID | 7.9 | 0.62 | 20 |
| 8 | 411 | Haaland | man city | FWD | 15.6 | 0.59 | 19 |
| 9 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.56 | 18 |
| 10 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.41 | 13 |
| 11 | 305 | Davis | ipswich | DEF | 4.0 | 0.41 | 13 |
| 12 | 124 | Groß | brighton | MID | 5.9 | 0.41 | 13 |
| 13 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.34 | 11 |
| 14 | 384 | Donnarumma | man city | GKP | 5.5 | 0.34 | 11 |
| 15 | 109 | Verbruggen | brighton | GKP | 4.5 | 0.34 | 11 |
| 16 | 200 | Lacroix | chelsea | DEF | 5.9 | 0.31 | 10 |
| 17 | 41 | Buendía | aston villa | MID | 5.9 | 0.31 | 10 |
| 18 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.28 | 9 |
| 19 | 588 | Khalaili | crystal palace | DEF | 5.0 | 0.28 | 9 |
| 20 | 173 | Thomas | coventry | DEF | 4.0 | 0.25 | 8 |
| 21 | 542 | E.Le Fée | sunderland | MID | 5.7 | 0.25 | 8 |
| 22 | 183 | Rudoni | coventry | MID | 4.9 | 0.25 | 8 |
| 23 | 286 | Belloumi | hull | MID | 5.1 | 0.25 | 8 |
| 24 | 154 | Palmer | chelsea | MID | 9.7 | 0.22 | 7 |
| 25 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.22 | 7 |
| 26 | 388 | Guéhi | man city | DEF | 6.0 | 0.22 | 7 |
| 27 | 567 | Horníček | newcastle | GKP | 5.0 | 0.22 | 7 |
| 28 | 597 | Suzuki | aston villa | GKP | 5.0 | 0.19 | 6 |
| 29 | 229 | Tarkowski | everton | DEF | 6.2 | 0.19 | 6 |
| 30 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.19 | 6 |
| 31 | 335 | Stach | leeds | MID | 6.0 | 0.19 | 6 |
| 32 | 334 | Muharemović | leeds | DEF | 5.0 | 0.16 | 5 |
| 33 | 385 | Trafford | leeds | GKP | 5.0 | 0.16 | 5 |
| 34 | 82 | Kelleher | brentford | GKP | 5.0 | 0.16 | 5 |
| 35 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.16 | 5 |
| 36 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.16 | 5 |
| 37 | 330 | Bogle | leeds | DEF | 4.6 | 0.16 | 5 |
| 38 | 1 | Raya | arsenal | GKP | 6.1 | 0.16 | 5 |
| 39 | 60 | Hill | bournemouth | DEF | 5.5 | 0.12 | 4 |
| 40 | 230 | Branthwaite | everton | DEF | 5.5 | 0.12 | 4 |
| … | (21 more) | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Low freq = path-sensitive / avoid locking capital.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20
```

