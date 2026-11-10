# Tabular squad card

Each row is one player in one fixture of the confirmed card. The
features are the header across the window and that player's minutes,
goals, and assists. The header is the team, the opponent, home or away,
and the Shin de-vigged home, draw, and away probabilities. The squad's
minutes and positions in the fixture being scored are on the same row.
A hidden event is the mask. Concede, regulars, own history, and
opponent history stay out.

The one-shot reader keeps every fixture. A cut hides later goals and
assists and leaves minutes, venue, opponents, and odds in place.
The stepped reader keeps the true past, keeps minutes, venue, and odds
on the next gameweek, blanks that week's opponent, and drops every
later column. Its loss is only that week.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

The gate, at 20% revealed, puts 190 goal bars on the same appearance
and 258 within one gameweek.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | one-shot | goals | 730 | 704 | 132 | 165 | 318 |
| 20% | one-shot | assists | 707 | 492 | 51 | 68 | 127 |
| 40% | one-shot | goals | 544 | 511 | 114 | 129 | 253 |
| 40% | one-shot | assists | 531 | 333 | 45 | 56 | 98 |
| 60% | one-shot | goals | 360 | 384 | 81 | 92 | 183 |
| 60% | one-shot | assists | 351 | 222 | 30 | 36 | 64 |
| 80% | one-shot | goals | 199 | 194 | 40 | 49 | 90 |
| 80% | one-shot | assists | 192 | 109 | 14 | 18 | 32 |
| 20% | stepped | goals | 730 | 868 | 198 | 236 | 429 |
| 20% | stepped | assists | 707 | 597 | 68 | 84 | 169 |
| 40% | stepped | goals | 544 | 673 | 161 | 181 | 344 |
| 40% | stepped | assists | 531 | 456 | 54 | 61 | 130 |
| 60% | stepped | goals | 360 | 455 | 112 | 124 | 230 |
| 60% | stepped | assists | 351 | 303 | 36 | 41 | 84 |
| 80% | stepped | goals | 199 | 232 | 64 | 71 | 122 |
| 80% | stepped | assists | 192 | 159 | 19 | 21 | 37 |

At 20% revealed the one-shot reader puts 132 goal bars on the same appearance
and 165 within one gameweek. It paints 704 goal bars against 730 actual.
The stepped reader puts 198 and 236, and paints 868.
The gate puts 190 and 258.
The gate stays the stripped card measured at 15:53.

Wall clock: 5879 seconds on CPU.

