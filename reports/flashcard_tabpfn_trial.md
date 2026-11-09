# Pretrained tabular reader

TabPFN v2 is a classifier pretrained on synthetic tables. Its public
checkpoint is the one used here. Each target gets a stratified sample
of 5,000 rows from 2022-23, 2023-24, and 2024-25, at the 20%, 40%,
60%, and 80% cuts, so the share of 0, 1, 2, and 3 stays as it is.
Concede, regulars, own history, and opponent history are not columns.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

The gate, at 20% revealed, puts 190 goal bars on the same appearance
and 258 within one gameweek.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | tabpfn | goals | 730 | 584 | 158 | 239 | 331 |
| 20% | tabpfn | assists | 707 | 596 | 88 | 146 | 194 |
| 40% | tabpfn | goals | 544 | 487 | 134 | 193 | 282 |
| 40% | tabpfn | assists | 531 | 475 | 87 | 130 | 178 |
| 60% | tabpfn | goals | 360 | 335 | 89 | 129 | 188 |
| 60% | tabpfn | assists | 351 | 325 | 64 | 97 | 129 |
| 80% | tabpfn | goals | 199 | 188 | 63 | 84 | 111 |
| 80% | tabpfn | assists | 192 | 179 | 31 | 49 | 62 |

At 20% revealed this reader puts 158 goal bars on the same appearance
and 239 within one gameweek. It paints 584 goal bars against 730 actual.
The gate puts 190 and 258. The gate stays the stripped card measured at 15:53.

Wall clock: 1602 seconds on CPU.

