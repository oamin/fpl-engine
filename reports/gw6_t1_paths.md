# ojaminFC Gameweek 6 paths

Two paths from the squad already owned. Both use the minutes file and the priced weeks only. The wildcard path is one rebuild, paid from the bank and sales. The hold path is the free-transfer search on those same weeks, with the hold margin 1.25 and the switch penalty 1.0. A week after the last priced week is not in the search. The figures are this plan's expected points for the priced weeks.

Minutes file SHA-256 prefix `c847b88986bee173`. Prices are millions of pounds. A sale uses the rules-module formula.

## Wildcard

Bank left £0.1m. Sells: Forster (GKP, BOU, 4.0), Lammens (GKP, MUN, 4.9), Calafiori (DEF, ARS, 5.7), Guéhi (DEF, MCI, 6.0), Shaw (DEF, MUN, 4.3), van Ewijk (DEF, COV, 4.0), Barnes (MID, NEW, 6.0), Cherki (MID, MCI, 7.6), Ødegaard (MID, ARS, 6.6), Calvert-Lewin (FWD, LEE, 6.0), Scarlett (FWD, TOT, 4.5). Buys: Raya (GKP, ARS, 6.1), Verbruggen (GKP, BHA, 4.5), Hall (DEF, NEW, 5.3), Murillo (DEF, NFO, 5.5), Silva (DEF, BOU, 5.0), Thomas (DEF, COV, 4.0), Buendía (MID, AVL, 5.9), Mbeumo (MID, MUN, 7.9), Rudoni (MID, COV, 4.9), Barry (FWD, EVE, 5.7), Wissa (FWD, NEW, 6.2).

| Player | Pos | Club | Price | GW6 xP | XI |
| --- | --- | --- | --- | --- | --- |
| Raya | GKP | ARS | 6.1 | 3.79 | XI |
| Verbruggen | GKP | BHA | 4.5 | 3.66 | bench |
| Davis | DEF | IPS | 4.0 | 4.67 | XI |
| Hall | DEF | NEW | 5.3 | 5.11 | XI |
| Murillo | DEF | NFO | 5.5 | 4.93 | XI |
| Thomas | DEF | COV | 4.0 | 4.83 | XI |
| Silva | DEF | BOU | 5.0 | 4.67 | bench |
| B.Fernandes | MID | MUN | 12.0 | 5.99 | XI |
| Mbeumo | MID | MUN | 7.9 | 6.15 | captain |
| Rogers | MID | CHE | 7.6 | 5.93 | XI |
| Rudoni | MID | COV | 4.9 | 4.95 | XI |
| Buendía | MID | AVL | 5.9 | 4.38 | bench |
| Barry | FWD | EVE | 5.7 | 5.18 | XI |
| Wissa | FWD | NEW | 6.2 | 4.79 | XI |
| Haaland | FWD | MCI | 15.5 | 4.59 | bench |

## Hold

Hits 0. Bank left £0.5m. Sells: van Ewijk (DEF, COV, 4.0). Buys: Silva (DEF, BOU, 5.0).

| Player | Pos | Club | Price | GW6 xP | XI |
| --- | --- | --- | --- | --- | --- |
| Lammens | GKP | MUN | 5.0 | 3.60 | XI |
| Forster | GKP | BOU | 4.0 | 0.00 | bench |
| Calafiori | DEF | ARS | 5.5 | 4.02 | XI |
| Davis | DEF | IPS | 4.0 | 4.67 | XI |
| Guéhi | DEF | MCI | 6.0 | 3.79 | XI |
| Silva | DEF | BOU | 5.0 | 4.67 | XI |
| Shaw | DEF | MUN | 4.5 | 3.36 | bench |
| B.Fernandes | MID | MUN | 12.0 | 5.99 | captain |
| Barnes | MID | NEW | 6.0 | 3.63 | XI |
| Cherki | MID | MCI | 7.5 | 3.53 | XI |
| Rogers | MID | CHE | 7.6 | 5.93 | XI |
| Ødegaard | MID | ARS | 6.5 | 3.51 | XI |
| Haaland | FWD | MCI | 15.5 | 4.59 | XI |
| Calvert-Lewin | FWD | LEE | 6.0 | 2.25 | bench |
| Scarlett | FWD | TOT | 4.5 | 0.00 | bench |

## Priced weeks

| Path | GW6 XI | GW6 bench | GW7 XI | GW7 bench |
| --- | --- | --- | --- | --- |
| Wildcard | 62.45 | 17.31 | 59.28 | 17.07 |
| Hold | 53.91 | 5.61 | 52.73 | 6.20 |
| Wildcard − hold | 8.54 | 11.70 | 6.55 | 10.86 |

## Chip plan

This is a `score_xp` run. There is no `slot_t1.json`, so it is not the `ep_next` decision.

`plan_half` plays the wildcard in Gameweek 6. The schedule also names Bench Boost in Gameweek 7. That slot is the better of the two priced weeks only. It is not a Gameweek 7 commitment. Triple Captain was already used in Gameweek 1. Free Hit is unused.

The wildcard clears its bar of 16 against the squad with no transfer: 62.45 − 52.61 in Gameweek 6 and 59.28 − 49.44 in Gameweek 7, together 19.68. Against the one-transfer path above (van Ewijk to Silva) the same rebuilt eleven leads by 8.54 + 6.55 = 15.09, which does not clear 16.

The plan value 138.80 is Gameweek 6 rebuilt eleven 62.45, plus Gameweek 7 rebuilt eleven 59.28, plus the Gameweek 7 rebuilt bench 17.07. Gameweek 8 and later add nothing, because only Gameweeks 6 and 7 have their own 1X2.

Gameweek 8 is an outright forecast (held eleven 51.95 on the current squad). Gameweek 9 repeats that row. It is not a new pot.

Haaland is on 45 minutes and scores 4.59. The wildcard benches him. Barry at 5.18 and Wissa at 4.79 start. That follows from the minutes file.

Score file: `data/predictions/2026-27/gw06/20261010T081519Z.csv` (207 players). Minutes `xmi_t1.csv`, prefix `c847b88986bee173`. Exchange lines `betfair_t1/gw_lines.csv`, prefix `98e5bb4281312953`.
