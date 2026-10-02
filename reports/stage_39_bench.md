# Stage 39 — bench weight inside transfers

One locked weight, `w = 0.25`, named `bench_w_ft`.
The opening squad, the fielded XI, the captain, automatic substitutes, and banked points stay on expected points.
Chips stay empty. The switch penalty stays outside transfer value, at 1.0.
The screen is 2025/26 gameweeks 5–38. The comparator is the paired expected-points climb on that same frame.
A clear loss is 100 or more points under that climb. Only then is 2024/25 skipped.
The pass bar of +34 is recorded on each season that is run. It is not a second weight.

A delta inside the 2025/26 opening-squad band of 1839 to 1972 is not an edge.

The weight can spend starting-XI money on the bench. It can take hits to fix bench players. All four bench slots are weighted the same.

## Screen 2025/26

| method | weight | points | vs xp | transfers | transfers/GW | hits |
|---|---:|---:|---:|---:|---:|---:|
| xp_ft | 0.00 | 1868 | +0 | 35 | 1.03 | 3 |
| bench_w_ft | 0.25 | 1926 | +58 | 39 | 1.15 | 7 |

Paired xp_ft is 1868 (inside the opening-squad band).
bench_w_ft is 1926 (inside the opening-squad band), delta +58.
Pass bar +34 on 2025/26: cleared.

## 2024/25

2024/25 was run. The screen was not a clear loss.

| method | weight | points | vs xp | transfers | transfers/GW | hits |
|---|---:|---:|---:|---:|---:|---:|
| xp_ft | 0.00 | 1889 | +0 | 42 | 1.24 | 9 |
| bench_w_ft | 0.25 | 1737 | -152 | 47 | 1.38 | 14 |

2024/25 delta -152. Pass bar +34 on 2024/25: missed.

## Diagnostics

Screen hits 3 to 7, transfers 35 to 39.
Screen points from substitutes 131 to 159. Unfilled XI slots after substitutes 8 to 2.
2024/25 hits 9 to 14, transfers 42 to 47, points from substitutes 146 to 107, unfilled XI slots after substitutes 11 to 13.
The pre-registered failure cases stay open. Extra hits are consistent with paying to fix bench players. The same weight on all four bench slots does not separate a playing substitute from a dead one. Nothing in the rule stops the search from spending starting-XI money on the bench.

## Result

Both 2025/26 totals sit in 1839 to 1972, so the +58 screen delta is not an edge.
bench_w_ft is the best of this batch, and the only method in it. It is not carried forward. The weight stays 0.25.

