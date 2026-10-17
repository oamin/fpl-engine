# ojaminFC Gameweek 6 team, score_xp

This is the selector the practice runs used. `score_xp` chooses the squad. `forecast_xp` fills Gameweek 8. `plan_half` searches the chip schedules. Minutes are the updated `xmi_t1.csv`, hash prefix `efb2c9b0cb1be3de`. Exchange lines are `betfair_t1/gw_lines.csv`, hash prefix `98e5bb4281312953`. Fixtures are `betfair_t1/fixtures.json`, hash prefix `477fee196153c242`. Prices are the T−1 bootstrap, hash prefix `f53c43b2c822d111`. It is not locked. `week_freeze.jsonl` was not appended.

Entry 2632584. Bank 15 tenths. One free transfer. Triple Captain was played in Gameweek 1. Chips still available before this run: wildcard, free hit, bench boost.

## Team this run names

Wildcard in Gameweek 6. Formation 4-4-2. Mbeumo is captain on 6.15. Fernandes is vice on 5.99. Haaland starts on 5.02, from 90 minutes. Bank left £0.5m. No hit.

The plan value is 138.38. That is the Gameweek 6 rebuilt eleven 63.14, the Gameweek 7 rebuilt eleven 58.22, and the Gameweek 7 rebuilt bench 17.01. Gameweek 8 and later add nothing to that sum.

The schedule also names Bench Boost in Gameweek 7. That is the larger of the two priced benches. It is read again when Gameweek 7 has its own line. Free Hit is unused. The Gameweek 6 free-hit eleven equals the rebuild, and the wildcard schedule is higher once Gameweek 7 is included.

| Player | Pos | Club | Price | GW6 xP | XI |
| --- | --- | --- | --- | ---: | --- |
| Raya | GKP | ARS | 6.1 | 3.79 | XI |
| Verbruggen | GKP | BHA | 4.5 | 3.66 | bench |
| Hall | DEF | NEW | 5.3 | 5.11 | XI |
| Khalaili | DEF | CRY | 5.0 | 5.13 | XI |
| Murillo | DEF | NFO | 5.5 | 4.93 | XI |
| Thomas | DEF | COV | 4.0 | 4.83 | XI |
| Davis | DEF | IPS | 4.0 | 4.67 | bench |
| B.Fernandes | MID | MUN | 11.9 | 5.99 | vice |
| Mbeumo | MID | MUN | 7.9 | 6.15 | captain |
| Rogers | MID | CHE | 7.8 | 5.93 | XI |
| Rudoni | MID | COV | 4.9 | 4.95 | XI |
| Mainoo | MID | MUN | 5.5 | 4.61 | bench |
| Barry | FWD | EVE | 5.7 | 5.18 | XI |
| Haaland | FWD | MCI | 15.6 | 5.02 | XI |
| Wissa | FWD | NEW | 6.2 | 4.79 | bench |

Price is the T−1 cost in £m. A sale uses the rules-module formula. Sells: Forster (GKP, BOU, 4.0), Lammens (GKP, MUN, 4.9), Calafiori (DEF, ARS, 5.7), Guéhi (DEF, MCI, 6.0), Shaw (DEF, MUN, 4.3), van Ewijk (DEF, COV, 4.0), Barnes (MID, NEW, 6.0), Cherki (MID, MCI, 7.6), Ødegaard (MID, ARS, 6.6), Calvert-Lewin (FWD, LEE, 6.0), Scarlett (FWD, TOT, 4.5). Buys: Raya (GKP, ARS, 6.1), Verbruggen (GKP, BHA, 4.5), Hall (DEF, NEW, 5.3), Khalaili (DEF, CRY, 5.0), Murillo (DEF, NFO, 5.5), Thomas (DEF, COV, 4.0), Mainoo (MID, MUN, 5.5), Mbeumo (MID, MUN, 7.9), Rudoni (MID, COV, 4.9), Barry (FWD, EVE, 5.7), Wissa (FWD, NEW, 6.2). Fernandes, Rogers, Haaland, and Davis stay.

The player column is one copy of `score_xp`. The eleven totals below count the captain a second time.

## Hold path

Hits 0. Bank left £0.5m. Sells van Ewijk (DEF, COV, 4.0). Buys Silva (DEF, BOU, 5.0).

| Player | Pos | Club | Price | GW6 xP | XI |
| --- | --- | --- | --- | ---: | --- |
| Lammens | GKP | MUN | 4.9 | 3.60 | XI |
| Forster | GKP | BOU | 4.0 | 0.00 | bench |
| Calafiori | DEF | ARS | 5.9 | 4.02 | XI |
| Davis | DEF | IPS | 4.0 | 4.67 | XI |
| Guéhi | DEF | MCI | 6.0 | 3.79 | XI |
| Silva | DEF | BOU | 5.0 | 4.67 | XI |
| Shaw | DEF | MUN | 4.3 | 3.36 | bench |
| B.Fernandes | MID | MUN | 11.9 | 5.99 | captain |
| Barnes | MID | NEW | 6.1 | 3.63 | XI |
| Cherki | MID | MCI | 7.8 | 3.53 | XI |
| Rogers | MID | CHE | 7.8 | 5.93 | vice |
| Ødegaard | MID | ARS | 6.8 | 3.50 | XI |
| Haaland | FWD | MCI | 15.6 | 5.02 | XI |
| Calvert-Lewin | FWD | LEE | 6.0 | 2.25 | bench |
| Scarlett | FWD | TOT | 4.5 | 0.00 | bench |

## Priced weeks

Gameweeks 6 and 7 have their own 1X2. Those are the weeks in the chip sum.

| Path | GW6 XI | GW6 bench | GW7 XI | GW7 bench |
| --- | --- | --- | ---: | ---: |
| Wildcard | 63.14 | 17.72 | 58.22 | 17.01 |
| Hold | 54.33 | 5.61 | 53.15 | 6.20 |
| Untouched | 53.02 | 4.46 | 50.28 | 4.88 |

The wildcard leads the untouched squad by 10.12 in Gameweek 6 and 7.94 in Gameweek 7, together 18.06, which clears 16. The same rebuild leads the van Ewijk to Silva path by 8.81 and 5.07, together 13.88.

## Gameweek 8

Gameweek 8 is filled from the outright strengths. It is not a copy of Gameweek 7. The largest player gap between those two weeks is 1.28. The held eleven is 52.38 and the rebuilt eleven is 59.29. Gameweek 9 repeats the Gameweek 8 row. It is not a new pot. Neither week is in the 138.38.

Haaland's Gameweek 6 score is 5.02. On the earlier minutes file, at 45 minutes, that score was 4.59 and the wildcard benched him. At 90 minutes he starts, and Wissa is the forward on the bench.

Score file: `data/predictions/2026-27/gw06/20261010T091116Z.csv` (Gameweeks 6, 7, and 8). Gemini accepted the reading ([score_xp team](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)).

## After a lock

When this team is confirmed, the lock log should store the entry, the wildcard fifteen, Mbeumo and Fernandes, the sell and buy lists, the wildcard chip, the four file hashes, and this note. That log is not written yet.
