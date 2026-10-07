Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

Two computations for 2022-23 both average −14.45. They are not the same weeks.

The opening-portfolio column is the frozen score_xp fifteen minus the frozen expected-points fifteen, with the XI chosen by expected points, and no transfer. The diverging column is each score building its own squad and then transferring. Gameweek 7 is absent from both.

| GW | opening portfolio | diverging squad |
|---:|---:|---:|
| 5 | -6 | +1 |
| 6 | -37 | -19 |
| 8 | -16 | -16 |
| 9 | -49 | -45 |
| 10 | +26 | -4 |
| 11 | +7 | +2 |
| 12 | +13 | +6 |
| 13 | -15 | -1 |
| 14 | -1 | -3 |
| 15 | -47 | -37 |
| 16 | -38 | -44 |
| 17 | +7 | -25 |
| 18 | +1 | -19 |
| 19 | -35 | -38 |
| 20 | -31 | -10 |
| 21 | -20 | -33 |
| 22 | -14 | +16 |
| 23 | -16 | -21 |
| 24 | +6 | -8 |
| 25 | -11 | -48 |
| 26 | -1 | +3 |
| 27 | -58 | -44 |
| 28 | -5 | +7 |
| 29 | -27 | -69 |
| 30 | -34 | -21 |
| 31 | -13 | -3 |
| 32 | -4 | -3 |
| 33 | -11 | -7 |
| 34 | -25 | +6 |
| 35 | +5 | +14 |
| 36 | +7 | -13 |
| 37 | -21 | -2 |
| 38 | -14 | +1 |

The two 2022–23 series (the opening-portfolio gap under the expected-points XI and the diverging squad gap) both average −14.45 points per gameweek and sum to −477, but agree on only 1 of 33 weeks and represent distinct processes.
The matching week is gameweek 8. The correlation of the two weekly series is +0.5838.
The 2022–23 deficit is distributed across the season, is present both before and after the World Cup hiatus, and is not caused by missing odds joins (0 unmatched of 26,505 rows).
That odds-join count is the one already printed in `reports/decision_placebo.md`. It was not recomputed.

Leave-one-season-out intervals measure the sensitivity of pooled estimates to individual season cohorts; they are diagnostic and do not replace the four-season results.

Each fold drops one season and resamples gameweeks inside each remaining season. The point estimate is the mean of the concatenated weeks. B=1000 and the seed is 0. A remaining season under 20 weeks is omitted. A conditional disagreement fold uses a minimum of 5 weeks instead, and that waiver is unchanged. A fold with fewer than two complete seasons is not identified. Concordance is not re-aggregated: the weekly difference is not a stored column.

The published four-season intervals remain the official evaluation benchmarks and are unchanged.
A sign flip or shift in interval bounds when holding out a season is evidence of cohort heterogeneity, not a justification to drop any season or declare a winner.

The published unconditional three-week interval is −1.69 [−2.92, −0.52]. About ten comparisons already share these four seasons, with no multiplicity control. That interval is suggestive and not conclusive. A leave-one-out interval is the same kind of evidence.

No new closed-season contrast is in this file. The certified likelihood was not re-aggregated.

### greedy score_xp − greedy expected points

Published four-season mean -1.6667. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +2.4706 [-0.8044, +5.3142] | the interval covers zero | sign differs |
| 2023-24 | 101 | -3.2475 [-6.7624, -0.1869] | the interval stays below zero | same sign |
| 2024-25 | 101 | -2.5941 [-6.2488, +0.7339] | the interval covers zero | same sign |
| 2025-26 | 101 | -3.3366 [-6.7525, -0.0193] | the interval stays below zero | same sign |

### shuffled greedy − shuffled hold

Published four-season mean +1.8444. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +2.3922 [-0.5007, +5.1772] | the interval covers zero | same sign |
| 2023-24 | 101 | -0.1782 [-2.6743, +2.2970] | the interval covers zero | sign differs |
| 2024-25 | 101 | +3.4059 [+0.7715, +6.2673] | the interval stays above zero | same sign |
| 2025-26 | 101 | +1.7525 [-0.7228, +4.4557] | the interval covers zero | same sign |

### greedy score_xp − shuffled greedy

Published four-season mean +15.0519. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +17.9804 [+14.7743, +21.7752] | the interval stays above zero | same sign |
| 2023-24 | 101 | +13.9010 [+10.7809, +17.0993] | the interval stays above zero | same sign |
| 2024-25 | 101 | +13.2178 [+9.6426, +16.6238] | the interval stays above zero | same sign |
| 2025-26 | 101 | +15.0792 [+11.4547, +18.6337] | the interval stays above zero | same sign |

### score_xp greedy − score_xp hold

Published four-season mean +7.3556. That interval was not recomputed. Strawman.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +7.4804 [+4.2451, +10.9020] | the interval stays above zero | same sign |
| 2023-24 | 101 | +5.2277 [+1.7790, +8.6438] | the interval stays above zero | same sign |
| 2024-25 | 101 | +8.2574 [+5.3262, +11.3270] | the interval stays above zero | same sign |
| 2025-26 | 101 | +8.4554 [+5.0884, +12.2084] | the interval stays above zero | same sign |

### common state, one-week score_xp − expected points

Published four-season mean -0.2901. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 99 | +0.0505 [-0.4747, +0.5457] | the interval covers zero | sign differs |
| 2023-24 | 98 | -0.5816 [-1.2862, +0.1939] | the interval covers zero | same sign |
| 2024-25 | 98 | -0.4184 [-1.1327, +0.3166] | the interval covers zero | same sign |
| 2025-26 | 98 | -0.2143 [-1.0102, +0.6327] | the interval covers zero | same sign |

### common state, three-week score_xp − expected points

Published four-season mean -1.6947. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 99 | -1.1616 [-2.5657, +0.0407] | the interval covers zero | same sign |
| 2023-24 | 98 | -1.7755 [-3.2962, -0.4589] | the interval stays below zero | same sign |
| 2024-25 | 98 | -1.8776 [-3.2449, -0.6018] | the interval stays below zero | same sign |
| 2025-26 | 98 | -1.9694 [-3.5719, -0.5196] | the interval stays below zero | same sign |

### common state, one-week score_xp − rolling three-week points

Published four-season mean +0.2672. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 99 | +0.7677 [-0.7376, +2.2333] | the interval covers zero | same sign |
| 2023-24 | 98 | +0.3163 [-0.9490, +1.5717] | the interval covers zero | same sign |
| 2024-25 | 98 | +0.0102 [-1.3676, +1.2452] | the interval covers zero | same sign |
| 2025-26 | 98 | -0.0306 [-1.5314, +1.4594] | the interval covers zero | sign differs |

### common state, one-week score_xp − shuffled score

Published four-season mean +3.9771. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 99 | +4.4040 [+3.1205, +5.7174] | the interval stays above zero | same sign |
| 2023-24 | 98 | +4.0612 [+2.5916, +5.5008] | the interval stays above zero | same sign |
| 2024-25 | 98 | +3.0204 [+1.8161, +4.2758] | the interval stays above zero | same sign |
| 2025-26 | 98 | +4.4184 [+3.0306, +5.8069] | the interval stays above zero | same sign |

### common state, one-week expected points − shuffled score

Published four-season mean +4.2672. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 99 | +4.3535 [+2.9692, +5.7583] | the interval stays above zero | same sign |
| 2023-24 | 98 | +4.6429 [+3.1020, +6.0921] | the interval stays above zero | same sign |
| 2024-25 | 98 | +3.4388 [+2.1630, +4.7247] | the interval stays above zero | same sign |
| 2025-26 | 98 | +4.6327 [+3.2334, +6.1026] | the interval stays above zero | same sign |

### opening fifteen, XI by score_xp, xp − exp

Published four-season mean -2.1852. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +0.5980 [-2.2174, +3.3632] | the interval covers zero | sign differs |
| 2023-24 | 101 | -3.3960 [-6.5943, -0.2864] | the interval stays below zero | same sign |
| 2024-25 | 101 | -2.5545 [-5.5946, +0.2683] | the interval covers zero | same sign |
| 2025-26 | 101 | -3.4158 [-6.6636, -0.4552] | the interval stays below zero | same sign |

### opening fifteen, XI by score_xp, xp − shuffled

Published four-season mean +7.7481. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +11.6863 [+7.9902, +15.6669] | the interval stays above zero | same sign |
| 2023-24 | 101 | +6.6733 [+3.3455, +10.5943] | the interval stays above zero | same sign |
| 2024-25 | 101 | +6.6931 [+2.9693, +10.1693] | the interval stays above zero | same sign |
| 2025-26 | 101 | +5.9010 [+1.9601, +9.7062] | the interval stays above zero | same sign |

### opening fifteen, XI by score_xp, xp − price ladder

Published four-season mean +4.6370. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +9.6275 [+6.0093, +13.1863] | the interval stays above zero | same sign |
| 2023-24 | 101 | +3.4653 [-0.5448, +7.2280] | the interval covers zero | same sign |
| 2024-25 | 101 | +1.5446 [-2.2082, +5.3072] | the interval covers zero | same sign |
| 2025-26 | 101 | +3.8614 [+0.4552, +7.3663] | the interval stays above zero | same sign |

### opening fifteen, XI by expected points, xp − exp

Published four-season mean -3.3037. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +0.3039 [-2.6181, +3.1478] | the interval covers zero | sign differs |
| 2023-24 | 101 | -5.2079 [-8.5260, -1.9797] | the interval stays below zero | same sign |
| 2024-25 | 101 | -3.4554 [-6.3970, -0.4547] | the interval stays below zero | same sign |
| 2025-26 | 101 | -4.8911 [-8.3270, -1.4849] | the interval stays below zero | same sign |

### opening fifteen, XI by expected points, xp − shuffled

Published four-season mean +7.7556. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +11.5294 [+7.8725, +15.2949] | the interval stays above zero | same sign |
| 2023-24 | 101 | +6.2772 [+2.6332, +10.2295] | the interval stays above zero | same sign |
| 2024-25 | 101 | +6.8416 [+3.2069, +10.2975] | the interval stays above zero | same sign |
| 2025-26 | 101 | +6.3366 [+2.2564, +10.3772] | the interval stays above zero | same sign |

### opening fifteen, XI by expected points, xp − price ladder

Published four-season mean +4.1778. That interval was not recomputed.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 102 | +9.8824 [+6.2824, +13.4414] | the interval stays above zero | same sign |
| 2023-24 | 101 | +2.8416 [-1.1916, +7.0797] | the interval covers zero | same sign |
| 2024-25 | 101 | +1.0099 [-2.8913, +5.0106] | the interval covers zero | same sign |
| 2025-26 | 101 | +2.9208 [-0.6344, +6.5351] | the interval covers zero | same sign |

### disagreement weeks, one-week score_xp − expected points

Published four-season mean -0.7451. That interval was not recomputed. Conditional: the 20-week floor does not apply.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 28 | +0.1786 [-1.5714, +2.0723] | the interval covers zero | sign differs |
| 2023-24 | 38 | -1.5000 [-3.3164, +0.4737] | the interval covers zero | same sign |
| 2024-25 | 41 | -1.0000 [-2.5366, +0.6341] | the interval covers zero | same sign |
| 2025-26 | 46 | -0.4565 [-2.0223, +1.2397] | the interval covers zero | same sign |

### disagreement weeks, three-week score_xp − expected points

Published four-season mean -4.3529. That interval was not recomputed. Conditional: the 20-week floor does not apply.

| held out | weeks | estimate | reading | versus published mean |
|---|---:|---|---|---|
| 2022-23 | 28 | -4.1071 [-8.3937, +0.0018] | the interval covers zero | same sign |
| 2023-24 | 38 | -4.5789 [-8.1849, -1.3151] | the interval stays below zero | same sign |
| 2024-25 | 41 | -4.4878 [-7.5128, -1.5854] | the interval stays below zero | same sign |
| 2025-26 | 46 | -4.1957 [-7.3696, -1.0859] | the interval stays below zero | same sign |

`greedy_xp_minus_hold_xp` is reported as the previously retired strawman baseline (a static squad that never transfers) and does not measure transfer skill.

Holding out 2022–23 shifts the three-week common-state interval to −1.1616 [−2.5657, +0.0407], which covers zero; the exclusion of zero in the published four-season estimate (−1.6947 [−2.9162, −0.5189]) is sensitive to that single cohort.
Without 2022–23, the greedy contrast point estimate is +2.4706 points per gameweek, but the interval [−0.8044, +5.3142] covers zero; setting that season aside provides no evidence that `score_xp` outperforms expected points.
No leave-one-out fold justifies dropping 2022–23 from the evaluation or promoting `score_xp` over expected points.
The published four-season pooled intervals remain the official benchmarks and are unchanged.
Both informed scores remain strictly separated from the within-week shuffle across all leave-one-season-out folds.

The pattern already reported on the disagreement weeks, cheaper buys, expected minutes about 77 against about 83, higher attack strength, and a naive buy that is often Haaland or Salah, is a hypothesis for the live weeks. It is not fit on the closed seasons.

A gain in the bulk of the list, such as who will not play, rather than among the best players, is a hypothesis. It is not a finding.

For live squad decisions from gameweek 6 onward, the primary score is pre-registered as `ep_next`, with `score_xp` logged alongside.
Promotion of `score_xp` over `ep_next` requires the paired live interval across at least 20 pre-deadline gameweeks to stay strictly above zero; an interval covering zero is undetermined and testing continues through gameweek 38.
The published historical score stays `score_xp`. The live scorer still prices the half with `score_xp`. This lock does not rewire it.

In accordance with protocol, if a paired transfer contrast interval covers zero, the result is inconclusive and no winner is declared.
No winner is declared. `score_xp` is unchanged.

Gemini kept the estimator and, after these folds, the reading that no fold promotes `score_xp` ([leave-one-season-out](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).
