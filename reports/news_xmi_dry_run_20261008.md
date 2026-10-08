# News minutes and a differential fifteen

Research dry runs for Gameweek 6, stamped 2026-10-08T19:59:49Z. The score stays `score_xp`. The picker opposite the string agent is the numeric engine: `price_half` plus a fresh squad MILP. These fifteens are new squads inside a £100.0m budget. They are not a one-free-transfer move from ojaminFC, and they are not the Saturday T−1h decision. `data/live/xmi_gw6.csv`, the 22:09 capture, and both freeze ledgers are unchanged.

Gemini accepted the three results together ([news dry run](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)). The best of this batch on `score_xp` is the unconstrained numeric fifteen.

| Run | XI + captain | Cost (tenths) | Captain |
| --- | ---: | ---: | --- |
| Numeric, news minutes | 67.33 | 996 | Saka |
| Numeric, ownership under 15% | 61.10 | 854 | Saka |
| String agent, ownership under 15% | 51.42 | 866 | Saka |

The string figure is a research overlay on the new score file. That column was not in the string prompt.

## News minutes

Minutes are `compile_player_xmi` (packets, then the last three gameweeks with zeros kept). The new score file is `data/predictions/2026-27/gw06/news_xmi_20261008T195949Z.csv` (207 rows). Forty-two players who already had a row in `xmi_gw6.csv` moved by more than 0.05.

Haaland goes from 90 (llm) to 45 (`50/50`). His `score_xp` goes from 4.74 to 4.44. He stays eligible and is not in the fresh fifteen. Wood's compiled minutes are 11.3 (`rolling avg`). He still has no `score_xp` row. Scarlett stays 0 (`transferred`). Shaw goes from 80 to 57.7. Palmer (id 154) goes from 65 to 45 (`50/50`). Saka was already in the 22:09 score file at 7.80 and was absent from the stored minutes file. The compile gives him 86.3 (`rolling avg`) and 7.57.

An `injured` tag with a chance between 0 and 100 writes `chance / 100` times the appearance prior. João Pedro is `d`, chance 75, tag `injured`, minutes 67.5, `score_xp` 4.73 (was 4.23 at 55 minutes). That clears the 45 gate, and the numeric engine starts him. Havertz is the same rule at 64.2 minutes. He is on the numeric differential bench. The string agent left him out after The Standard's line that he is out with a hamstring injury.

## Numeric fifteen

Formation 4-5-1. XI 59.77. With Saka counted twice, 67.33.

| Player | Role | score_xp | Own % | xmi | Tag |
| --- | --- | ---: | ---: | ---: | --- |
| Verbruggen | XI | 3.66 | 21.6 | 90.0 | rolling avg |
| Gabriel | XI | 5.27 | 22.7 | 90.0 | rolling avg |
| Calafiori | XI | 5.10 | 50.5 | 82.0 | rolling avg |
| Hall | XI | 4.99 | 17.4 | 89.7 | rolling avg |
| Thomas | XI | 4.94 | 8.5 | 90.0 | rolling avg |
| Saka | XI, captain | 7.57 | 13.8 | 86.3 | rolling avg |
| Rogers | XI, vice | 6.22 | 40.5 | 88.7 | rolling avg |
| Mbeumo | XI | 6.19 | 20.5 | 90.0 | rolling avg |
| B.Fernandes | XI | 6.02 | 38.5 | 90.0 | rolling avg |
| Rudoni | XI | 5.08 | 0.1 | 69.0 | rolling avg |
| João Pedro | XI | 4.73 | 65.0 | 67.5 | injured |
| Martinez | bench | 3.61 | 4.7 | 90.0 | rolling avg |
| Murillo | bench | 4.85 | 2.1 | 89.7 | rolling avg |
| Barry | bench | 4.72 | 8.3 | 90.0 | rolling avg |
| Wissa | bench | 4.67 | 18.7 | 90.0 | rolling avg |

Arsenal has three (Gabriel, Calafiori, Saka). Newcastle has Hall and Wissa. Manchester United has Mbeumo and B.Fernandes.

## Numeric differential

The owned-player bypass is off. Everyone needs three prior appearances, minutes at least 45, and `selected_by_percent` strictly under 15. Missing ownership is out. The pool is 179 (GKP 20, DEF 73, MID 72, FWD 14). The MILP found a fifteen.

Formation 5-4-1. XI 53.53. With Saka counted twice, 61.10. That is 6.23 under the unconstrained fifteen. Haaland, Calafiori, Rogers, B.Fernandes, and João Pedro are out on ownership.

| Player | Role | score_xp | Own % | xmi | Tag |
| --- | --- | ---: | ---: | ---: | --- |
| Martinez | XI | 3.61 | 4.7 | 90.0 | rolling avg |
| Thomas | XI | 4.94 | 8.5 | 90.0 | rolling avg |
| Murillo | XI | 4.85 | 2.1 | 89.7 | rolling avg |
| Lacroix | XI | 4.78 | 6.8 | 61.3 | rolling avg |
| Davis | XI | 4.73 | 6.9 | 90.0 | rolling avg |
| Silva | XI | 4.47 | 0.4 | 90.0 | rolling avg |
| Saka | XI, captain | 7.57 | 13.8 | 86.3 | rolling avg |
| Rudoni | XI, vice | 5.08 | 0.1 | 69.0 | rolling avg |
| Buendía | XI | 4.47 | 1.0 | 75.3 | rolling avg |
| E.Le Fée | XI | 4.30 | 3.4 | 90.0 | rolling avg |
| Barry | XI | 4.72 | 8.3 | 90.0 | rolling avg |
| Lammens | bench | 3.61 | 9.9 | 90.0 | rolling avg |
| Gonzalo | bench | 4.04 | 4.0 | 81.7 | rolling avg |
| Havertz | bench | 4.04 | 8.2 | 64.2 | injured |
| Tielemans | bench | 4.16 | 1.5 | 82.0 | rolling avg |

## String differential

The context listed 639 players under 15% and no score column. The model returned a legal fresh fifteen: cost 866, bank 134, no chip, empty transfer lists, formation 3-4-3. Captain Saka, vice Dewsbury-Hall. Highest ownership in the squad is Saka at 13.8, then Schade 11.6, Thiago 9.1, van Ewijk 8.7. Fulham and Everton are at the club cap of three.

| Player | Role | Own % | score_xp |
| --- | --- | ---: | ---: |
| Martinez | XI | 4.7 | 3.61 |
| Branthwaite | XI | 2.4 | 4.13 |
| Van de Ven | XI | 3.8 | 2.23 |
| Bassey | XI | 0.7 | 2.85 |
| Saka | XI, captain | 13.8 | 7.57 |
| Barnes | XI | 5.8 | 3.57 |
| Dewsbury-Hall | XI, vice | 5.9 | 3.83 |
| Schade | XI | 11.6 | 3.50 |
| Thiago | XI | 9.1 | 3.80 |
| Barry | XI | 8.3 | 4.72 |
| Gonzalo | XI | 4.0 | 4.04 |
| E.Le Fée | bench | 3.4 | 4.30 |
| Lindelöf | bench | 0.3 | 3.64 |
| van Ewijk | bench | 8.7 | 3.02 |
| Leno | bench | 3.4 | 3.59 |

XI with Saka counted twice is 51.42. That is 9.68 under the numeric differential and 15.91 under the unconstrained fifteen. Van de Ven is in the string eleven on a could-return note. His `score_xp` is 2.23. The agent also left out Havertz, Saliba, Scott, and Damsgaard after the filed press lines.

## What Gemini asked to keep visible

- An injured tag with a partial chance still clears 45 minutes, so João Pedro starts and Havertz makes the differential bench.
- The string eleven can take a low `score_xp` defender (Van de Ven, 2.23) from a return note.
- All three squads are fresh budgets, not the held squad's next transfer.
- The live submission stays the Saturday T−1h slot.
