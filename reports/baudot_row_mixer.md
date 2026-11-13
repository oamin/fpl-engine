# Baudot row-card block-token mixer (full past + one-shot remainder)

Exploratory trial (not a gated score comparison).

## Card format

- GW as **rows** (left→right bits).
- Header: club 8 · opp 8 · odds 12 (assessed win ON · draw BLANK · opp win ON).
- Player: role 4 + id **11** + mins 6 + G 3 + A 3.
- Player IDs: unique by normalised name across closed seasons (1670 names; ID_BITS=11).

## Model / FOV

Block-token mixer sees **all past GW rows** (true events). Remainder rows keep club/opp/odds/who/mins with **G/A blank**. One forward paints the **entire remainder event window**. Loss = Soft-Dice + 0.05 BCE on that window only. Cuts: 20%, 40%, 60%, 80%. No μ/share/argmax.

Train seasons: 2022-23, 2023-24, 2024-25 (60 club cards). Test: 2025-26 (20 cards).

## Holdout — volume

| channel | actual bars | painted bars | paint ratio | GKP painted |
|---|---:|---:|---:|---:|
| goals | 2016 | 2902 | 1.44× | 0 |
| assists | 1883 | 3044 | 1.62× | — |

## Holdout — blanks vs real events (pooled cuts)

| channel | blank cells | kept blank | false paint | event cells | exact on events | within ±1 on events |
|---|---:|---:|---:|---:|---:|---:|
| goals | 38284 | 36092 | 2192 | 1833 | 602 (0.33) | 1764 (0.96) |
| assists | 38336 | 35797 | 2539 | 1781 | 454 (0.25) | 1730 (0.97) |

## By cut

| cut | channel | actual | painted | event exact rate | event within±1 |
|---|---|---:|---:|---:|---:|
| 20% | goals | 805 | 958 | 0.28 | 0.96 |
| 20% | assists | 749 | 1014 | 0.22 | 0.96 |
| 40% | goals | 598 | 905 | 0.34 | 0.97 |
| 40% | assists | 558 | 941 | 0.26 | 0.98 |
| 60% | goals | 394 | 658 | 0.38 | 0.96 |
| 60% | assists | 372 | 687 | 0.30 | 0.97 |
| 80% | goals | 219 | 381 | 0.39 | 0.96 |
| 80% | assists | 204 | 402 | 0.29 | 0.98 |

Final train loss: 0.7637.

Artifacts: `/opt/cursor/artifacts/baudot_season_arsenal.png`, `baudot_teach_diagram.png`.
