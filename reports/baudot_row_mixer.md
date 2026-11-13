# Baudot row-card block-token mixer (full past + one-shot remainder)

Exploratory trial (not a gated score comparison).

Train: 2022-23, 2023-24, 2024-25 (60 cards). Test: 2025-26 (20 cards).
FOV: all past GW rows true; remainder G/A blank; one-shot paint of the remainder window.
IDs: 1670 unique names, ID_BITS=11. Metrics match `tally_row` + `balanced_score`.

| revealed | channel | actual events | painted events | hits | actual ±1 | painted ±1 | score |
|---|---|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 958 | 244 | 320 | 490 | 0.220 |
| 20% | assists | 707 | 1014 | 172 | 213 | 343 | 0.139 |
| 40% | goals | 544 | 905 | 219 | 281 | 443 | 0.182 |
| 40% | assists | 531 | 941 | 153 | 177 | 287 | 0.117 |
| 60% | goals | 360 | 658 | 157 | 197 | 316 | 0.169 |
| 60% | assists | 351 | 687 | 116 | 136 | 217 | 0.114 |
| 80% | goals | 199 | 381 | 90 | 111 | 170 | 0.162 |
| 80% | assists | 192 | 402 | 64 | 79 | 117 | 0.103 |

Hits = same player-GW appearance. ±1 = event GW matched within one gameweek (actual→painted / painted→actual). Score = hit F1 × volume ratio (`balanced_score` in `flashcard_card_score.py`).

GKP painted goal events by cut: 20%=0, 40%=0, 60%=0, 80%=0.

Final train loss: 0.7637.

Artifacts: `/opt/cursor/artifacts/baudot_season_arsenal.png`, `baudot_teach_diagram.png`.
