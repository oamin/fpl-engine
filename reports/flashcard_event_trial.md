# Taller event bars

Goals and assists each use a sixteen-pixel band, the same height as
the minute bar. One event fills five pixels, two fill eleven, and three
fill the band. A hidden event is still the whole band in grey.
The first reader is the locked linear patch reader. The second freezes
ImageNet ResNet-18 through its first residual stage and trains the
coupler on those features. Each cell is stretched to a square first,
so the chart is not resized into one photograph.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.

The previous gate, at 20% revealed, puts 190 goal bars on the same
appearance and 258 within one gameweek.

| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---|---|---:|---:|---:|---:|---:|
| 20% | taller | goals | 730 | 740 | 189 | 249 | 383 |
| 20% | taller | assists | 707 | 678 | 100 | 158 | 221 |
| 40% | taller | goals | 544 | 576 | 143 | 192 | 296 |
| 40% | taller | assists | 531 | 523 | 77 | 114 | 163 |
| 60% | taller | goals | 360 | 376 | 93 | 132 | 195 |
| 60% | taller | assists | 351 | 339 | 55 | 82 | 108 |
| 80% | taller | goals | 199 | 200 | 55 | 71 | 99 |
| 80% | taller | assists | 192 | 181 | 34 | 50 | 63 |
| 20% | imagenet | goals | 730 | 661 | 182 | 241 | 364 |
| 20% | imagenet | assists | 707 | 601 | 89 | 147 | 195 |
| 40% | imagenet | goals | 544 | 513 | 140 | 187 | 286 |
| 40% | imagenet | assists | 531 | 463 | 70 | 118 | 154 |
| 60% | imagenet | goals | 360 | 335 | 90 | 123 | 177 |
| 60% | imagenet | assists | 351 | 300 | 52 | 87 | 104 |
| 80% | imagenet | goals | 199 | 176 | 49 | 66 | 92 |
| 80% | imagenet | assists | 192 | 160 | 34 | 51 | 63 |

At 20% revealed the taller linear reader puts 189 goal bars on the
same appearance and 249 within one gameweek. The ImageNet reader
puts 182 and 241, and paints 661 goal bars against 730 actual.
Neither replaced the gate of that day, which was 190 and 258. The gate is now the square-card full painter. The other squad is not an input.

Legend: `data/plots/flashcard_event_bars.png`.
One 2025-26 card, top eight players and first twelve fixtures, at 20% revealed:
`data/plots/flashcard_event_card.png`.

Wall clock: 200 seconds on CPU.

