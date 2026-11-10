# Pretrained tabular reader

Each row is one cell of the stripped card. The cell carries the player
slot, the fixture slot, position, minutes, the sheet flag, venue, and
the club's recent scoring rate. Goal and assist tokens are the painted
counts, or the mask on a hidden cell. That fixture's forty squad rows
and forty opponent rows are on the same row, as position and minutes.
Club and opponent use a fixed code, the same number in every season.
Concede, regulars, own history, and opponent history are absent.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

The gate, at 20% revealed, puts 190 goal bars on the same appearance
and 258 within one gameweek.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | tabpfn | goals | 730 | 1477 | 274 | 349 | 605 |
| 20% | tabpfn | assists | 707 | 1568 | 210 | 304 | 483 |
| 40% | tabpfn | goals | 544 | 1556 | 250 | 309 | 551 |
| 40% | tabpfn | assists | 531 | 1429 | 177 | 240 | 407 |
| 60% | tabpfn | goals | 360 | 1091 | 175 | 219 | 383 |
| 60% | tabpfn | assists | 351 | 1006 | 118 | 163 | 267 |
| 80% | tabpfn | goals | 199 | 680 | 109 | 131 | 214 |
| 80% | tabpfn | assists | 192 | 795 | 95 | 109 | 201 |

At 20% revealed this reader puts 274 goal bars on the same appearance
and 349 within one gameweek. It paints 1477 goal bars against 730 actual.
The gate puts 190 and 258. The gate stays the stripped card measured at 15:53.

Wall clock: 8508 seconds on CPU.

