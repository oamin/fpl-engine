# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **10** coherent strands: each fixture samples λ from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), Poisson scoreline, path rate update.
- Player xP: frozen deadline shares/xmi/defcon (`add_player_priors`) × scenario pot via `xp_on_pot` (score_xp v2). Share seeding: **False**.
- Snapshots at GW [8, 10, 12, 14, 16, 18] → selectable-pool MILP (`xi_plus_captain`).
- Solved squads: **60** · selectable **180** (170 with nonzero share) · seed 7.

Scoring: `Opta match_prediction λ bands → sampled scenario pot; deadline add_player_priors shares/xmi/defcon × xp_on_pot (score_xp v2)`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 12 | Saka | arsenal | MID | 9.6 | 0.85 | 0.85 | 0.48 | 51 |
| 2 | 572 | Tzolakis | hull | GKP | 4.7 | 0.82 | 0.00 | 0.00 | 49 |
| 3 | 40 | Rogers | chelsea | MID | 7.8 | 0.73 | 0.73 | 0.12 | 44 |
| 4 | 566 | Silva | bournemouth | DEF | 5.0 | 0.63 | 0.63 | 0.07 | 38 |
| 5 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.62 | 0.02 | 0.00 | 37 |
| 6 | 449 | Hall | newcastle | DEF | 5.3 | 0.60 | 0.60 | 0.02 | 36 |
| 7 | 411 | Haaland | man city | FWD | 15.6 | 0.55 | 0.55 | 0.12 | 33 |
| 8 | 249 | Barry | everton | FWD | 5.7 | 0.53 | 0.53 | 0.00 | 32 |
| 9 | 427 | Mbeumo | man utd | MID | 7.9 | 0.53 | 0.53 | 0.10 | 32 |
| 10 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.53 | 0.53 | 0.00 | 32 |
| 11 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.43 | 0.18 | 0.00 | 26 |
| 12 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.42 | 0.42 | 0.02 | 25 |
| 13 | 305 | Davis | ipswich | DEF | 4.0 | 0.42 | 0.32 | 0.03 | 25 |
| 14 | 534 | Hume | sunderland | DEF | 4.3 | 0.42 | 0.00 | 0.00 | 25 |
| 15 | 345 | Tanaka | leeds | MID | 4.9 | 0.38 | 0.00 | 0.00 | 23 |
| 16 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.37 | 0.37 | 0.00 | 22 |
| 17 | 464 | Wissa | newcastle | FWD | 6.2 | 0.32 | 0.32 | 0.05 | 19 |
| 18 | 109 | Verbruggen | brighton | GKP | 4.5 | 0.30 | 0.30 | 0.00 | 18 |
| 19 | 200 | Lacroix | chelsea | DEF | 5.9 | 0.28 | 0.28 | 0.00 | 17 |
| 20 | 194 | Thomas-Asante | coventry | FWD | 5.0 | 0.25 | 0.00 | 0.00 | 15 |
| 21 | 4 | Gabriel | arsenal | DEF | 8.0 | 0.25 | 0.25 | 0.00 | 15 |
| 22 | 588 | Khalaili | crystal palace | DEF | 5.0 | 0.25 | 0.25 | 0.00 | 15 |
| 23 | 154 | Palmer | chelsea | MID | 9.7 | 0.23 | 0.23 | 0.00 | 14 |
| 24 | 173 | Thomas | coventry | DEF | 4.0 | 0.23 | 0.15 | 0.00 | 14 |
| 25 | 279 | Ajayi | hull | DEF | 4.2 | 0.23 | 0.00 | 0.00 | 14 |
| 26 | 335 | Stach | leeds | MID | 6.0 | 0.23 | 0.23 | 0.00 | 14 |
| 27 | 267 | Lukić | ipswich | MID | 4.9 | 0.23 | 0.00 | 0.00 | 14 |
| 28 | 124 | Groß | brighton | MID | 5.9 | 0.18 | 0.18 | 0.00 | 11 |
| 29 | 330 | Bogle | leeds | DEF | 4.6 | 0.18 | 0.18 | 0.00 | 11 |
| 30 | 250 | Leno | fulham | GKP | 4.5 | 0.15 | 0.00 | 0.00 | 9 |
| 31 | 542 | E.Le Fée | sunderland | MID | 5.7 | 0.13 | 0.13 | 0.00 | 8 |
| 32 | 41 | Buendía | aston villa | MID | 5.9 | 0.13 | 0.13 | 0.00 | 8 |
| 33 | 388 | Guéhi | man city | DEF | 6.0 | 0.13 | 0.13 | 0.00 | 8 |
| 34 | 15 | Ødegaard | arsenal | MID | 6.8 | 0.13 | 0.13 | 0.00 | 8 |
| 35 | 226 | Pickford | everton | GKP | 5.5 | 0.12 | 0.12 | 0.00 | 7 |
| 36 | 384 | Donnarumma | man city | GKP | 5.5 | 0.12 | 0.12 | 0.00 | 7 |
| 37 | 567 | Horníček | newcastle | GKP | 5.0 | 0.12 | 0.12 | 0.00 | 7 |
| 38 | 286 | Belloumi | hull | MID | 5.1 | 0.12 | 0.12 | 0.00 | 7 |
| 39 | 229 | Tarkowski | everton | DEF | 6.2 | 0.10 | 0.10 | 0.00 | 6 |
| 40 | 60 | Hill | bournemouth | DEF | 5.5 | 0.10 | 0.10 | 0.00 | 6 |
| … | (39 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain
```

