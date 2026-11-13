# Flashcard gate — Baudot learnable Tversky (locked 2026-11-13)

This is the **default flashcard painter gate**. New flashcard trials compare here.

- Module: `src.models.flashcard_baudot_row_trial`
- FOV: all_past_gws + one_shot_remainder
- Loss: learnable Tversky (β≥α); locked checkpoint α=0.447, β=0.553
- Seed: 0; ID_BITS=11
- Metrics: `tally_row` + `balanced_score` (`flashcard_card_score.GATE`)

| revealed | channel | actual events | painted events | hits | actual ±1 | painted ±1 | score |
|---|---|---:|---:|---:|---:|---:|---:|
| 20% | goals | 730 | 701 | 194 | 268 | 377 | 0.260 |
| 40% | goals | 544 | 557 | 151 | 214 | 299 | 0.268 |
| 60% | goals | 360 | 383 | 105 | 145 | 205 | 0.266 |
| 80% | goals | 199 | 195 | 60 | 82 | 101 | 0.298 |
| 20% | assists | 707 | 693 | 114 | 153 | 232 | 0.160 |
| 40% | assists | 531 | 551 | 89 | 123 | 174 | 0.159 |
| 60% | assists | 351 | 377 | 66 | 90 | 126 | 0.169 |
| 80% | assists | 192 | 191 | 32 | 45 | 59 | 0.166 |

GKP painted goal events: 0 at every cut.

Supersedes the square-card full-painter gate of 2026-11-12 (`GATE_SQUARE_2026_11_12` in `flashcard_card_score.py`).
