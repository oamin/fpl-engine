# Baudot row-card — learnable Tversky (full past + one-shot remainder)

Exploratory trial (not a gated score comparison).

Train: 2022-23, 2023-24, 2024-25 (60 cards). Test: 2025-26 (20 cards).
FOV: all past GW rows true; remainder G/A blank; one-shot paint of the remainder window.
Loss: **learnable Tversky** with α=0.5−δ, β=0.5+δ, δ=0.49·σ(logit) ⇒ **β≥α**, α+β=1. Logit starts at −2 (near Dice); cosine LR + early-stop on holdout mean score.
Learned weights at best checkpoint: α=0.447, β=0.553 (epochs run 20).
IDs: 1670 unique names, ID_BITS=11. Metrics: `tally_row` + `balanced_score`.

| revealed | channel | actual events | painted events | hits | actual ±1 | painted ±1 | score |
|---|---|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 701 | 194 | 268 | 377 | 0.260 |
| 20% | assists | 707 | 693 | 114 | 153 | 232 | 0.160 |
| 40% | goals | 544 | 557 | 151 | 214 | 299 | 0.268 |
| 40% | assists | 531 | 551 | 89 | 123 | 174 | 0.159 |
| 60% | goals | 360 | 383 | 105 | 145 | 205 | 0.266 |
| 60% | assists | 351 | 377 | 66 | 90 | 126 | 0.169 |
| 80% | goals | 199 | 195 | 60 | 82 | 101 | 0.298 |
| 80% | assists | 192 | 191 | 32 | 45 | 59 | 0.166 |

Hits = same player-GW appearance. ±1 = event GW matched within one gameweek. Score = hit F1 × volume ratio.

GKP painted goal events by cut: 20%=0, 40%=0, 60%=0, 80%=0.

Final train loss: 0.7714.
Best holdout mean score during train: 0.218.

Artifacts: `/opt/cursor/artifacts/baudot_season_arsenal.png`, `baudot_teach_diagram.png`.
