# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **20** coherent Monte Carlo strands: each GW samples scorelines inside Opta λ bounds, updates club GF/GA path.
- Snapshots at GW [8, 10, 12, 14, 16, 18]: score season selectable pool (`xp_on_pot` on path pots) → squad MILP.
- Solved squads: **120** · selectable **180** · seed 7.

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | n |
|---:|---:|---|---|---|---:|---:|---:|
| 1 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.72 | 87 |
| 2 | 305 | Davis | ipswich | DEF | 4.0 | 0.70 | 84 |
| 3 | 12 | Saka | arsenal | MID | 9.6 | 0.70 | 84 |
| 4 | 427 | Mbeumo | man utd | MID | 7.9 | 0.64 | 77 |
| 5 | 366 | Wirtz | liverpool | MID | 7.3 | 0.63 | 76 |
| 6 | 249 | Barry | everton | FWD | 5.7 | 0.62 | 75 |
| 7 | 411 | Haaland | man city | FWD | 15.6 | 0.59 | 71 |
| 8 | 464 | Wissa | newcastle | FWD | 6.2 | 0.48 | 58 |
| 9 | 106 | Thiago | brentford | FWD | 7.8 | 0.44 | 53 |
| 10 | 109 | Verbruggen | brighton | GKP | 4.5 | 0.42 | 51 |
| 11 | 40 | Rogers | chelsea | MID | 7.8 | 0.42 | 50 |
| 12 | 449 | Hall | newcastle | DEF | 5.3 | 0.41 | 49 |
| 13 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.39 | 47 |
| 14 | 124 | Groß | brighton | MID | 5.9 | 0.32 | 38 |
| 15 | 588 | Khalaili | crystal palace | DEF | 5.0 | 0.30 | 36 |
| 16 | 330 | Bogle | leeds | DEF | 4.6 | 0.30 | 36 |
| 17 | 388 | Guéhi | man city | DEF | 6.0 | 0.29 | 35 |
| 18 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.29 | 35 |
| 19 | 88 | Kayode | brentford | DEF | 4.6 | 0.27 | 32 |
| 20 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.27 | 32 |
| 21 | 279 | Ajayi | hull | DEF | 4.2 | 0.24 | 29 |
| 22 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.23 | 28 |
| 23 | 176 | Dasilva | coventry | DEF | 4.0 | 0.23 | 27 |
| 24 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.22 | 26 |
| 25 | 567 | Horníček | newcastle | GKP | 5.0 | 0.21 | 25 |
| 26 | 112 | Van Hecke | spurs | DEF | 4.9 | 0.21 | 25 |
| 27 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.21 | 25 |
| 28 | 566 | Silva | bournemouth | DEF | 5.0 | 0.20 | 24 |
| 29 | 204 | Mitchell | crystal palace | DEF | 4.5 | 0.20 | 24 |
| 30 | 183 | Rudoni | coventry | MID | 4.9 | 0.20 | 24 |
| 31 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.16 | 19 |
| 32 | 37 | Lindelöf | aston villa | DEF | 4.5 | 0.15 | 18 |
| 33 | 597 | Suzuki | aston villa | GKP | 5.0 | 0.15 | 18 |
| 34 | 591 | Enciso | ipswich | MID | 5.5 | 0.15 | 18 |
| 35 | 529 | Roefs | sunderland | GKP | 4.9 | 0.15 | 18 |
| 36 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.15 | 18 |
| 37 | 154 | Palmer | chelsea | MID | 9.7 | 0.14 | 17 |
| 38 | 542 | E.Le Fée | sunderland | MID | 5.7 | 0.14 | 17 |
| 39 | 82 | Kelleher | brentford | GKP | 5.0 | 0.14 | 17 |
| 40 | 173 | Thomas | coventry | DEF | 4.0 | 0.13 | 16 |
| … | (36 more) | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Low freq = path-sensitive / avoid locking capital.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20
```

