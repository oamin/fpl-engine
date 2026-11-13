# Baudot row-card block-token mixer

Exploratory trial (not a gated score comparison).

## Card format

- GW as **rows** (left→right bits).
- Header: club 8 · opp 8 · odds 12 (assessed win ON · draw BLANK · opp win ON).
- Player: role 4 + id **11** + mins 6 + G 3 + A 3.
- Player IDs: unique by normalised name across closed seasons (1670 names; 10 bits are not enough, so ID_BITS=11).

## Model

Block-token mixer: schema slices → temporal attention over past 5 GWs → squad attention on the target week → soft event ink. Loss = channel Soft-Dice + 0.05 BCE. No μ/share/argmax.

Train seasons: 2022-23, 2023-24, 2024-25 (60 club cards). Test: 2025-26 (20 cards).

## Holdout tallies (threshold 0.5)

| channel | actual bars | painted bars | exact row-matches | within 1 | exact rate |
|---|---:|---:|---:|---:|---:|
| goals | 1005 | 2928 | 16698 | 19549 | 0.853 |
| assists | 942 | 2881 | 16577 | 19556 | 0.847 |

GKP painted goals (should stay near 0): **0**.

Final train loss: 0.8009.

Artifacts: `/opt/cursor/artifacts/baudot_season_arsenal.png`, `baudot_teach_diagram.png`.
