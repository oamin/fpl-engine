# Decision margin

A plan, locked on 2026-10-06 before either histogram is read. Gemini amended it ([decision margin](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)). Nothing here changes `score_xp`, the hold of 1.25, the wildcard margin of 16, the free-hit margin of 12, or the formation list. ojaminFC stays out of every mean. Entry 1078627 stays out. No Odds API call.

## What the proposal gets right

The five-week gap is not another formation rule, another hold, or another ownership term. Those screens have already lost or stopped.

The carry from each manager's Gameweek 1 fifteen is −27.43, and 0 of 14 finish ahead. The lineup piece is −198, of which shared-squad shape is −196 and automatic substitutes are +15. That shape splits into formation −163 and displacement −33, and midfield is −132 of the formation. Forcing five midfielders then failed on the four earlier seasons. The shape is a description. It is not the lever.

The transfer piece is −180. Chip signings are −329, ranked-lower pairs are −81, and the model's other unshared starters are +230. Ranked-lower means the human already had the higher `score_xp`. Fourteen of the seventeen price pairs have no sale that pays for them, and a second sale funds six pairs that sum to −7. The cohort chain stopped there.

The chip weeks are the remaining mass. Of 85 signings, 83 were already in the buy pool. The rebuild held 19 of them, −100. It left 64 out, −229. At the median, that human's `score_xp` sat 0.31 below the rebuild's lowest player at the same position, and 46 of the 64 sat below that player. Club or price blocked 0.71 of those outside points. The three-appearance rule blocked none.

Outside expected points do not rescue the score. On the same 14 squads, Onside and official `ep_next` sit further below the points than `score_xp`.

## Where the proposal goes past the files

The lineup gap and the transfer gap are different players. The −196 is someone both fifteens owned and only one side started. The −329 is someone the human bought with a wildcard or a free hit. A ranking story has to say which of those it means. The one that matches "the model had its own player ahead" is the 64 left out of the rebuild, not the formation count and not the ranked-lower pairs.

There is no pair, among shared-squad starters at the same position, where the model had the lower score. Ranked-behind is 0. The −17 ranked-ahead figure is the higher score losing on the week, and −39 of those points are weeks it scored fewer. That is already a small reversal. It is not the −180.

The close-pair reading, on the four seasons before this one, does not show the leader losing when two eligible players sit within half a point. The win rates are 48.6%, 56.1%, 45.8%, and 56.6%. The haul-rate arm of that reading is parked. A winner's curse on the upper tail is a different table, and it has not been counted. It is not a reason to reopen the haul arm.

A file of elite human squads for the earlier seasons does not exist, and the entry API cannot build one. "Humans chose B" cannot be asked historically. The 14 are the sample.

A role-change flag taken from later minutes, or from anything learned after the deadline, is leakage. The news tags cover the live window and are mostly an ask. They do not become a residual on `score_xp`.

The residual `f()` on the decision boundary is not the next build. Five weeks are too few to fit it, and the label it would need is the one the files do not have.

## Reading A

The population is the 64 chip signings the rebuild left out. The pair is the rebuild's lowest `score_xp` at that position, the pair already used for the median of −0.31. No new alternative is chosen. The margin is that player's `score_xp` minus the human's. Positive means the model player was ahead.

The 64 are split by the block tag already stored. Constrained means club or price. Unconstrained means neither. The published constraint share is 0.71 of the −229, which is already at least half, so the top-level call on the full outside set is constraint. That call is not re-litigated.

The ranking call uses only the unconstrained rows. Their weight is the sum of the stored points on those rows. The bins are margin at or below 0, above 0 up to 0.5, above 0.5 up to 1.25, and above 1.25. A bin's share is its weight divided by the unconstrained total.

The ranking call is small-margin when the share above 0 and up to 0.5 is at least 0.5. It is wide-margin when the share above 1.25 is at least 0.5. Anything else is inconclusive. The threshold stays 0.5. The 19 players the rebuild held stay in their own row and out of the bins.

Beside the bar, and not part of it: human points minus the paired player's points, both players' expected minutes and actual minutes, the human's prior appearances in bands 0–2, 3–5, and 6 or more, and a news tag only where `news_tags_gw15` already has one that is not an ask. A player on 0 minutes stays in the pair. Actual minutes do not move a bin or a call. The pair-point difference is not the −229, and it is not described as that figure.

Neither call changes a constant. Inconclusive leaves the cohort chain stopped. Small-margin names the historical rank table as the description of a narrow disagreement. Wide-margin names a forecast miss on those unconstrained names, still without a new feature.

## Reading B

Locked in the same batch. It does not wait for Reading A.

Seasons 2022/23 through 2025/26. Weeks 6 through 38. An eligible row has at least three prior appearances and expected minutes of at least 45, the same cut as the close-pair reading. Within a position, players are ranked by `score_xp`. The bands are 1–5, 6–10, 11–20, and the rest. Bias is mean `score_xp` minus mean points. The comparison is that season's bias on every eligible row.

A winner's-curse claim needs both of these in every season: the top band's bias is positive, and it exceeds the all-eligible bias by at least 0.25. Anything else parks the claim. One season does not set a haircut. The claim does not change the score and does not reopen the haul-rate arm.

## After the two readings

The counts come next, from the rows already stored for Reading A and from the historical score table for Reading B. They are not in this note. A third reading, a residual added to `score_xp`, is not opened by either call.
