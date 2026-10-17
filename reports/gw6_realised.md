# Gameweek 6 realised

Evaluated 2026-10-14T16:22:48Z. Points are FPL `total_points` from the finished, data-checked Gameweek 6 live feed. Each eleven is the locked starting eleven with the captain counted twice. Automatic substitutes are not applied, so the totals match the freeze forecast rule.

Post-match sheet: `data/holdout/2026-27/player_gw/gw06.csv` (667 rows, sha256 `b342dada2b88cf6b0ca45c9e5cb1d6efbc3acdf8dd8e478bc43cfce758876a77`). Player `value` is the bootstrap cost at collection time. The Gameweek 1–5 holdout files are unchanged.

ojaminFC submitted the numagent wildcard. Entry score 56, bench 16, bank 5 tenths, one free transfer into Gameweek 7. Free hit and bench boost remain.

## Numagent (`numagent-gw6-lock`)

| Path | Actual XI+C | `ep_next` forecast | `score_xp` forecast |
| --- | ---: | ---: | ---: |
| Decision (wildcard) | 56 | 56.0 | 63.14 |
| Untouched hold | 51 | 49.5 | 53.02 |
| One transfer (van Ewijk → Silva) | 48 | 51.5 | 54.33 |

Net vs hold **+5**. Net vs one transfer **+8**. The primary bar metric for this week is decision − 1FT = **+8**.

`ep_next` on the locked wildcard eleven was exact. `score_xp` overshot that eleven by 7.14. Pool Spearman of Gameweek 6 `score_xp` against actual points on the 213-player score file is **0.331**. Clean-sheet Brier on the twenty club sides from the T−1 Exchange 1X2 and 2.5 line is **0.137**.

### Decision eleven

| Player | Pts | |
| --- | ---: | --- |
| Raya | 3 | |
| Hall | 0 | |
| Khalaili | 2 | |
| Murillo | 2 | |
| Thomas | 4 | |
| B.Fernandes | 1 | |
| Mbeumo | 8 | captain (16) |
| Rogers | 9 | |
| Rudoni | 7 | |
| Barry | 6 | |
| Haaland | 6 | |

Mbeumo, Rogers, Rudoni, Barry, and Haaland carried the rebuild. Hall blanked from 90 minutes. Verbruggen scored 9 on the bench; that bench was not in the eleven total.

### Counterfactuals

The held Gameweek 5 fifteen, Fernandes captain, scored 51. Cherki 10 and Calafiori 9 were the haul. The one-transfer path bought Silva (0) for van Ewijk and scored 48. Shaw, who stayed on the hold path, scored 3. That transfer cost three points against the hold.

## Stragent (`stragent-gw6`)

Scarlett to Muniz, no chip, Fernandes captain, Haaland vice. Actual XI+C **49**.

| Player | Pts | |
| --- | ---: | --- |
| Lammens | 2 | |
| Calafiori | 9 | |
| Davis | 1 | |
| Guéhi | 6 | |
| Barnes | 2 | |
| B.Fernandes | 1 | captain (2) |
| Rogers | 9 | |
| Cherki | 10 | |
| Ødegaard | 1 | |
| Haaland | 6 | |
| Muniz | 1 | |

Cherki and Calafiori paid the hold-style squad. Fernandes captain returned 2. Muniz, the only buy, scored 1. That eleven is 7 behind the submitted wildcard and 2 behind the untouched hold.

## Reading

The wildcard beat both of its own counterfactuals. The gap to the one-transfer path is the bar’s primary week total this week. `ep_next` named the wildcard eleven’s points exactly; `score_xp` ranked the pool better than chance (ρ 0.33) and still overshot the chosen eleven. Stragent kept most of the old fifteen, caught Cherki and Calafiori, and left seven points on the table against the rebuild. The first formal bar read stays at Gameweek 26.
