# Multistage MC + MILP — long-horizon selection frequency

Trial (diagnostic). Objective: **`xi_plus_captain_ft`**. Not a gated likelihood promotion. Chips off.

## Design

- Decision GW **6**; roll fixtures GW8–19.
- **8** coherent strands: priced weeks use **cascade_short** (`priced`={'gws': [8], 'n_fits': 10, 'n_props_matched': 330, 'odds_path': '/workspace/data/scratch/gw08_odds_full.json'}); unpriced weeks use **cascade_long** Opta place→GF/GA±MAE → `M` → `k ∼ M`.
- Player xP: **GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→E[G/A|k] share_mean→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW** (props=True; generative=True). Share seeding: **False**.
- Minutes overlay: **False** ({'overlay': False, 'n': 180}).
- Snapshots at GW [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] → selectable-pool MILP (`xi_plus_captain_ft`).
- Solved squads: **96** · selectable **180** (59 with nonzero rG) · seed 7.

Scoring: `GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→E[G/A|k] share_mean→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW`

## Player stability (selection frequency)

| rank | id | name | team | pos | £ | freq | xi | cap | n |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 572 | Tzolakis | hull | GKP | 4.7 | 1.00 | 0.31 | 0.00 | 96 |
| 2 | 449 | Hall | newcastle | DEF | 5.3 | 0.94 | 0.88 | 0.07 | 90 |
| 3 | 229 | Tarkowski | everton | DEF | 6.2 | 0.91 | 0.91 | 0.06 | 87 |
| 4 | 176 | Dasilva | coventry | DEF | 4.0 | 0.90 | 0.68 | 0.11 | 86 |
| 5 | 377 | Munoz | liverpool | MID | 6.5 | 0.89 | 0.88 | 0.41 | 85 |
| 6 | 68 | Tavernier | bournemouth | MID | 6.1 | 0.78 | 0.75 | 0.07 | 75 |
| 7 | 346 | Calvert-Lewin | leeds | FWD | 6.0 | 0.73 | 0.00 | 0.00 | 70 |
| 8 | 165 | João Pedro | chelsea | FWD | 7.7 | 0.73 | 0.62 | 0.00 | 70 |
| 9 | 86 | Lewis-Potter | brentford | MID | 5.5 | 0.73 | 0.71 | 0.08 | 70 |
| 10 | 110 | Rushworth | coventry | GKP | 4.5 | 0.69 | 0.45 | 0.00 | 66 |
| 11 | 411 | Haaland | man city | FWD | 15.6 | 0.67 | 0.58 | 0.00 | 64 |
| 12 | 569 | Gonzalo | fulham | FWD | 6.0 | 0.66 | 0.00 | 0.00 | 63 |
| 13 | 124 | Groß | brighton | MID | 5.9 | 0.60 | 0.59 | 0.04 | 58 |
| 14 | 330 | Bogle | leeds | DEF | 4.6 | 0.47 | 0.18 | 0.00 | 45 |
| 15 | 112 | Van Hecke | spurs | DEF | 4.9 | 0.46 | 0.36 | 0.01 | 44 |
| 16 | 286 | Belloumi | hull | MID | 5.1 | 0.45 | 0.43 | 0.04 | 43 |
| 17 | 534 | Hume | sunderland | DEF | 4.3 | 0.39 | 0.00 | 0.00 | 37 |
| 18 | 453 | Barnes | newcastle | MID | 6.1 | 0.33 | 0.33 | 0.01 | 32 |
| 19 | 12 | Saka | arsenal | MID | 9.6 | 0.32 | 0.31 | 0.02 | 31 |
| 20 | 426 | B.Fernandes | man utd | MID | 11.9 | 0.32 | 0.31 | 0.02 | 31 |
| 21 | 366 | Wirtz | liverpool | MID | 7.3 | 0.28 | 0.28 | 0.00 | 27 |
| 22 | 115 | De Cuyper | brighton | DEF | 5.0 | 0.24 | 0.21 | 0.00 | 23 |
| 23 | 204 | Mitchell | crystal palace | DEF | 4.5 | 0.21 | 0.20 | 0.03 | 20 |
| 24 | 305 | Davis | ipswich | DEF | 4.0 | 0.14 | 0.11 | 0.00 | 13 |
| 25 | 8 | Calafiori | arsenal | DEF | 5.9 | 0.12 | 0.11 | 0.00 | 12 |
| 26 | 279 | Ajayi | hull | DEF | 4.2 | 0.11 | 0.08 | 0.00 | 11 |
| 27 | 41 | Buendía | aston villa | MID | 5.9 | 0.10 | 0.09 | 0.00 | 10 |
| 28 | 40 | Rogers | chelsea | MID | 7.8 | 0.09 | 0.08 | 0.01 | 9 |
| 29 | 79 | Evanilson | bournemouth | FWD | 6.0 | 0.08 | 0.05 | 0.00 | 8 |
| 30 | 385 | Trafford | leeds | GKP | 5.0 | 0.07 | 0.06 | 0.00 | 7 |
| 31 | 597 | Suzuki | aston villa | GKP | 5.0 | 0.05 | 0.04 | 0.00 | 5 |
| 32 | 350 | A.Becker | liverpool | GKP | 5.5 | 0.05 | 0.03 | 0.00 | 5 |
| 33 | 1 | Raya | arsenal | GKP | 6.1 | 0.05 | 0.05 | 0.00 | 5 |
| 34 | 166 | N.Jackson | aston villa | FWD | 6.5 | 0.05 | 0.03 | 0.00 | 5 |
| 35 | 472 | Murillo | nottm forest | DEF | 5.5 | 0.03 | 0.02 | 0.00 | 3 |
| 36 | 384 | Donnarumma | man city | GKP | 5.5 | 0.03 | 0.02 | 0.00 | 3 |
| 37 | 109 | Verbruggen | brighton | GKP | 4.5 | 0.03 | 0.01 | 0.00 | 3 |
| 38 | 194 | Thomas-Asante | coventry | FWD | 5.0 | 0.03 | 0.00 | 0.00 | 3 |
| 39 | 138 | Kostoulas | brighton | FWD | 5.6 | 0.03 | 0.01 | 0.00 | 3 |
| 40 | 268 | King | fulham | MID | 5.6 | 0.02 | 0.02 | 0.00 | 2 |
| … | (16 more) | | | | | | | | |

## Read

High ``freq`` across strands × snapshots = appears in MILP squads under many coherent future paths → candidate for long-horizon hold. Under ``xi_plus_captain``, ``freq_captain`` is the share of solved squads where that player is the named captain.

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective xi_plus_captain_ft
```

