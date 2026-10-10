# ojaminFC Gameweek 6 team

This is the selector run. It is not locked. Nothing has been appended to `week_freeze.jsonl`.

The choice column is the T−1 `ep_next` capture, `official_20261010T082509Z_t1.csv`, stamp 2026-10-10T08:25:09Z. Minutes are `xmi_t1.csv`, hash prefix `efb2c9b0cb1be3de`. Exchange lines are `betfair_t1/gw_lines.csv`, hash prefix `98e5bb4281312953`. Prices are the same-stamp bootstrap. `score_xp` does not choose the squad. With `ep_next` passed in, only Gameweek 6 is priced. Later weeks add nothing.

Entry 2632584. Bank 15 tenths. One free transfer. Triple Captain was played in Gameweek 1. Chips still available: wildcard, free hit, bench boost. `GW6_CHIP_COUNTS` is false, so this judgement is not a chip-rule result.

## Submitted team

No chip. No transfer. No hit. The plan value is 60.5, which is this week's eleven with the captain counted twice. It is not a half-season total.

Formation 3-5-2. Davis is captain because his `ep_next` is 8.0 and Haaland's is 7.5. Haaland is vice.

| Player | Pos | Club | ep_next | XI |
| --- | --- | --- | ---: | --- |
| Lammens | GKP | MUN | 3.0 | XI |
| Forster | GKP | BOU | 0.0 | bench |
| Calafiori | DEF | ARS | 3.5 | XI |
| Davis | DEF | IPS | 8.0 | captain |
| Guéhi | DEF | MCI | 5.0 | XI |
| Shaw | DEF | MUN | 1.0 | bench |
| van Ewijk | DEF | COV | 0.0 | bench |
| Barnes | MID | NEW | 5.5 | XI |
| B.Fernandes | MID | MUN | 2.0 | XI |
| Cherki | MID | MCI | 4.5 | XI |
| Rogers | MID | CHE | 5.0 | XI |
| Ødegaard | MID | ARS | 2.5 | XI |
| Calvert-Lewin | FWD | LEE | 6.0 | XI |
| Haaland | FWD | MCI | 7.5 | vice |
| Scarlett | FWD | TOT | 0.0 | bench |

## Why no chip

On this one-week horizon the free-hit eleven is defined as the wildcard rebuild, so the two chips tie. The rule plays neither. The rebuild was not submitted. A separate transfer search was not the chip plan. Gemini accepted that reading ([ep_next team](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)).

## After a lock

When this team is confirmed, the lock log should store the entry, the fifteen above, the captain and vice, the empty transfer list, the empty chip, the four file hashes, the `ep_next` stamp, and this note. That log is not written yet.
