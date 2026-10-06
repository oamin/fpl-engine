# Chip horizon

A plan, locked on 2026-10-06 before any discounted wildcard sum is read. Gemini kept it ([chip horizon](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)). Nothing here changes `score_xp`, the hold of 1.25, the wildcard margin of 16, the free-hit margin of 12, or the formation list. The published chip choice stays the undiscounted sum. ojaminFC stays out. Entry 1078627 stays out. No Odds API call.

## Formula

The weight is the transfer-search `GAMMA` of 0.9. It is imported. It is not a second constant.

A priced window is the decision week plus up to two later club weeks. A blank is already skipped, and it does not use up a power. The discounted wildcard sum is γ to the power h, times that step's rebuilt eleven minus the held eleven. h is 0 on the decision week. One step has weight 1. Two steps use 1 and 0.9. Three use 1, 0.9, and 0.81. The weight multiplies the gap. The copied tail is not a step, and it is not added.

Free hit, bench boost, and triple captain stay as they are. The transfer value is not discounted again. The hurdle stays 16. There is no rescaled bar.

## What is compared

On each deadline the same chip rule is called twice. The second call discounts only the wildcard sum. A flip is a different chip. The weeks after a flip are not re-solved. `plan_half` is not edited. The empty-chip climb is not rerun. The twelve crowd climbs stay out: they decided on the whole half, and this batch does not reconstruct that path.

## Population

The 14 cohort managers, Gameweeks 1–5, on the free wallet and the path already carried. The live Gameweek 6 check uses the stored leads of 12.56 and 5.11. Their sum is 17.67, and the discounted figure is written only after that addition holds. No new search is run for that week.

## Failure cases

A negative later step raises the discounted sum, and it can newly clear 16. A positive later step can only lower it. A raw sum of 16 that sits entirely on the third step becomes 12.96 and misses. The same sum on the decision week is unchanged. A window of one or two steps is not padded. The free-hit hurdle of 12 is not discounted. Gameweek 1 cannot play a wildcard or a free hit. A tie within a millionth of a point still plays nothing.

The count is not in this note.
