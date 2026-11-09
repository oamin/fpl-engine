# Card comparison

Test: 2025-26, every club, every row, appearances after gameweek 19.
The reader is squad vision with squad coupling. Each future fixture is given
a number of bars equal to the sum of its bar probabilities. Those bars go to
the rows with minutes and the highest probabilities.

A bar matches on the same row only. Within one gameweek includes the same
appearance and either neighboring gameweek. The numbers are totals.

| channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |
|---|---:|---:|---:|---:|---:|
| goals | 459 | 440 | 119 | 160 | 232 |
| assists | 445 | 386 | 51 | 95 | 110 |

Of the 459 actual goal bars, 119 are the same appearance
and 160 have a painted bar within one gameweek.
The extra week adds 41 actual goal bars.
232 painted goal bars sit within one gameweek of an actual bar
on their row. One goal can have more than one painted bar inside that window.

Of the 445 actual assist bars, 51 are the same appearance
and 95 have a painted bar within one gameweek.
110 painted assist bars sit within one gameweek of an actual bar.

Wall clock: 76 seconds on CPU.

