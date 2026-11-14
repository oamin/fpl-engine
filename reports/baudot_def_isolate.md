# Baudot DEF isolate — train/test positions independently

Same mixer / learnable Tversky / FOV / seed as the locked gate. Only the **player window** on the card and in the loss changes.

- Train cards: 60 · holdout: 20 (2025-26)
- Gate: `src.models.flashcard_baudot_row_trial` · locked mean 0.218
- Early-stop metric: **DEF-only** mean balanced score (not full roster mean).

## Arms

| arm | picture | loss window |
|---|---|---|
| `1_def_loss_mask` | full roster (MID/FWD visible) | DEF cells only |
| `2_def_card_isolate` | non-DEF player blocks zeroed + padded out of squad attn | DEF cells only |

## DEF scoreboard (primary)

| arm | DEF mean score | DEF goals 20% a/p/h | DEF assists 20% a/p/h | α | β | epochs |
|---|---:|---|---|---:|---:|---:|
| gate (autopsy) | — | 103/0/0 | 181/0/0 | 0.447 | 0.553 | — |
| 1_def_loss_mask | 0.051 | 103/192/12 | 181/322/27 | 0.450 | 0.550 | 28 |
| 2_def_card_isolate | 0.051 | 103/0/0 | 181/226/25 | 0.446 | 0.554 | 17 |

## Soft ink on DEF (20% remainder)

| arm | channel | mean soft event | mean soft blank | mean max event | % max≥0.5 | n max≥0.05 |
|---|---|---:|---:|---:|---:|---:|
| gate (autopsy) | goals | 0.0063 | 0.0062 | 0.0098 | 0.000 | 0 |
| gate (autopsy) | assists | 0.0067 | 0.0065 | 0.0104 | 0.000 | 0 |
| 1_def_loss_mask | goals | 0.0471 | 0.0178 | 0.1394 | 0.117 | 35 |
| 1_def_loss_mask | assists | 0.0514 | 0.0208 | 0.1526 | 0.149 | 68 |
| 2_def_card_isolate | goals | 0.0766 | 0.0350 | 0.2217 | 0.000 | 77 |
| 2_def_card_isolate | assists | 0.0931 | 0.0408 | 0.2713 | 0.138 | 146 |

## Full position split at 20% (actual / painted / hits)

### Goals

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| gate (autopsy) | 0/0/0 | 103/0/0 | 394/~370/83 | 233/~331/111 |
| 1_def_loss_mask | 0/108/0 | 103/192/12 | 394/258/41 | 233/28/9 |
| 2_def_card_isolate | 0/0/0 | 103/0/0 | 394/0/0 | 233/0/0 |

### Assists

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| gate (autopsy) | 4/0/0 | 181/0/0 | 434/~363/65 | 88/~330/49 |
| 1_def_loss_mask | 4/167/2 | 181/322/27 | 434/357/63 | 88/52/9 |
| 2_def_card_isolate | 4/0/0 | 181/226/25 | 434/0/0 | 88/0/0 |

## DEF-only cut table (best arm by DEF mean)

Best arm: `1_def_loss_mask` (DEF mean 0.051).

| revealed | channel | actual | painted | hits | score |
|---|---|---:|---:|---:|---:|
| 20% | goals | 103 | 192 | 12 | 0.044 |
| 20% | assists | 181 | 322 | 27 | 0.060 |
| 40% | goals | 74 | 151 | 11 | 0.048 |
| 40% | assists | 140 | 250 | 23 | 0.066 |
| 60% | goals | 49 | 104 | 8 | 0.049 |
| 60% | assists | 91 | 174 | 14 | 0.055 |
| 80% | goals | 27 | 59 | 4 | 0.043 |
| 80% | assists | 46 | 94 | 6 | 0.042 |

## Morning call

**Partial / open.** Isolating DEF moves soft ink off the floor and lands non-trivial hits. Not a gate replacement (full-roster mean is undefined under DEF-only paint); keep as a rare-event diagnostic and decide whether to merge a DEF specialist into the shared painter.

Disposition: diagnostic keep — not promoted over the locked full-roster gate.

Read order:
1. Does DEF soft max leave the ~0.01 floor (n max≥0.05 > 0)?
2. Do DEF goals/assists leave `0/0` with real hits (not volume-only)?
3. Is `2_def_card_isolate` better than `1_def_loss_mask`? (If yes, the shared picture comparison was part of the collapse.)
4. Do not promote over the locked gate on DEF-only mean; a specialist would need a merge plan that preserves MID/FWD.
