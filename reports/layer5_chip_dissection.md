# L5 calendar climb — chip dissection (from joint path)

Source: `data/processed/layer5_calendar_vs_l4.json` (joint L5_cal vs L4).

## Headline

| slice | n GW | mean Δ vs L4 | sum Δ |
| --- | ---: | ---: | ---: |
| chip play weeks | 32 | **+3.56** | **+114** |
| none (transfer) weeks | 92 | **−4.49** | **−413** |

Chips are not losing on the weeks they are played. The loss is the **aftermath** (squad/FT path after early chip use).

## By chip on play weeks only

| chip | n | mean Δ | sum Δ |
| --- | ---: | ---: | ---: |
| free_hit | 8 | +14.0 | +112 |
| triple_captain | 8 | +1.8 | +14 |
| bench_boost | 8 | 0.0 | 0 |
| wildcard | 8 | −1.5 | −12 |

FH looks good in-week; WC slightly negative in-week. Neither explains −413 on none weeks.

## Timing

Every season burns a transfer chip by GW11 (WC or FH), often with TC in the same early window. Hits on none weeks: 288 vs 32 on chip weeks.

## Ablation (aborted)

Started single-chip ablation; **stopped** when `no_chips` itself was ~−120/season vs L4 on 2022-23/2023-24. That is not a chip result — `no_chips` was the L5 **beam** (myopic weekly transfers), not L4’s horizon MILP.

**First-order:** `src/strategy/layer5_nochips_diagnose.py` (2×2: horizon MILP vs beam × priced vs calendar).
