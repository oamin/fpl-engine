# FPL engine — one gameweek, for an outside reader

Paste this whole note. It is the published decision path. The score is `score_xp`. The chip map is empty, so no chip is played. Transfers stay in the same position. Nothing below is a suggestion. It is what the code does.

## What is carried in from last week

- A squad of 15 players, each with a purchase price.
- Money left in the bank.
- A free-transfer count, from 0 to 5.

## Constants used on this path

| Name | Value | What it does |
|---|---|---|
| Score | `score_xp` | Shot share times the team goal rate from 1X2 and totals. Appearance chance is already inside it. |
| Horizon | 3 playable weeks | This week, the next week that has matches, and the one after that. |
| Discount | 0.9, then 0.81 | Later weeks count less. |
| Hold margin | 1.25 | A sale must beat keeping the 15 by at least this much. |
| Switch penalty | 1.0 per transfer | Charged inside the value, on top of any hit. |
| Hit | 4 points | One hit per transfer beyond the free-transfer count. At most 2 hits are considered. |
| Transfers considered | at most 3 | And only `free transfers + 2`, if that is smaller. |
| Squad | 2 GK, 5 DEF, 5 MID, 3 FWD | Budget £100m at the first week. At most 3 from one club. |
| Buyable | last 3 appearances average at least 45 minutes | Anyone already owned stays in the pool even below that. |

`score_xp` is not rebuilt during the week. The lineup and the sales both use the score that already exists at the deadline.

## Step 1 — Is there a match to play?

Look at the season sheet for this gameweek.

- If no club has a row, skip the week. Do not sell, do not pick an XI, do not change the free-transfer count. Carry the same 15 into the next week. Stop.
- If at least one club has a row, continue.

The only fully empty week in the cache is 2022/23 gameweek 7.

## Step 2 — Who can be bought or kept

The pool is:

- every player whose last three appearances average at least 45 minutes, and
- every player already in the 15.

An owned player who has no row on this week's sheet keeps the latest score from a week before this deadline. His points and his minutes for this week are 0. A score from a later week is never used.

## Step 3 — Who has a fixture

- A club with at least one row on the sheet keeps its score.
- A club with no row is ranked at 0 for this week and for every later week in the three-week window.

A player who is benched while his club plays is not zeroed. Only a missing club is zeroed. A player on 0 whose club does play outranks a player on 0 whose club does not, when the XI is chosen.

## Step 4 — First week, or a normal week

**First week of the climb.** Build a free 15. Maximise the sum of all fifteen scores, under £100m, with 2/5/5/3 and at most three from one club. No hits. The free-transfer count becomes 1 after this week. Skip to step 7.

**Every later week.** Price the current 15 against candidate sales. Continue at step 5.

## Step 5 — What a sale is worth

Over the three playable weeks:

`V = this week's XI + 0.9 × the next XI + 0.81 × the one after − 4 × hits − 1 × transfers`

- The XI inside V is the best legal shape from that candidate 15, using the same score rule as step 7.
- Later weeks reuse this week's score. They are not re-forecast.
- A later week in which that player has no roster row scores 0.
- A week with no clubs is dropped, so it does not use up one of the three slots.
- A double gameweek is still scored once. There is no fixture multiplier.
- Hits are the transfers beyond the free-transfer count. A move of 1 when the count is 1 or more is 0 hits.

## Step 6 — Which sales are tried

Only a same-position swap is legal here. A defender is sold for a defender, and the same for the other positions.

1. Try up to 35 single swaps, taken from the largest raw score gaps, then re-priced with V.
2. From the best 6 of those, try one more swap.
3. Try a third swap only if two swaps is the current leader.

Stop at three transfers, and stop if a candidate would need a third hit.

Then:

- If the held 15 is no longer legal (a club change has put four players from one club in the squad), take the best legal 15 the search found. If it found none, the old 15 stays.
- If the held 15 is legal, make the move only when its V beats the hold by at least 1.25. Otherwise keep the 15.

The penalty and the hits are already inside V. The 1.25 is an extra hurdle on top of them.

## Step 7 — Pick the XI, the captain, and the bench

The 15 is now fixed.

For each allowed shape, take the highest score in each position. Keep the shape whose scores sum highest. The shapes on offer are 3-4-3, 3-5-2, 4-4-2, 4-3-3, 4-5-1, 5-3-2, and 5-4-1. The legal shape 5-2-3 is not offered, so older published totals stay comparable.

- Captain: highest score in that XI.
- Vice-captain: second highest score in that XI.
- Bench: the other four. The goalkeeper is first. The other three follow score order, highest first.

The bench adds nothing to V. It is whoever the XI did not use.

## Step 8 — After the matches

This scores the week. It does not change the 15 that was chosen at the deadline.

- A starter with 0 minutes is replaced by the next bench player who played, if the shape stays legal (1 goalkeeper, 3 to 5 defenders, 2 to 5 midfielders, 1 to 3 forwards).
- A bench player who did not play is skipped.
- A bench player who would break the shape is skipped and is not used up. The walk continues to the next bench player.
- Double the captain's points. If the captain played 0 minutes and the vice-captain played, double the vice-captain instead. If both played 0, there is no double.
- Subtract 4 for each hit.

The week total is the final XI, plus the captain extra, minus the hits.

## Step 9 — Free transfers for next week

- Spare free transfers roll forward, then one is added, and the count is capped at 5.
- A hit spends the count down to 0 before that extra one is added.
- The first week of the climb sets the count to 1 after the free 15.
- A skipped empty week does not change the count.

Carry the 15, the purchase prices, the bank, and the new count into the next gameweek.

A sale price, used when a swap is checked for budget, is half the rise rounded down, or the full fall if the price dropped.

## Chips

Closed on this path. A chip runs only when a separate week map names one.

- Wildcard and Free Hit rebuild the 15 from the bank plus sell prices. The moves are free. The free-transfer count is not spent and is not increased.
- Free Hit puts the previous 15 back after the week is scored.
- Bench Boost adds the points of the four who finished outside the XI.
- Triple Captain makes the captain extra three times the player's score, with the same vice-captain fall-through.

## What this path does not do

- It does not know that a player is injured, doubtful, or suspended. Those fields are not on the historical sheet in a form that was known before the deadline.
- It does not treat a 0-minute week as an injury.
- It does not reorder the bench to keep a defender in reserve. An illegal substitute is already skipped.
- It does not give the bench a weight in the transfer value.
- It does not change formation, the hold margin, the switch penalty, or `score_xp` from week to week.

## Figure

The same path is drawn in `data/plots/gw_decision_flow.png`. Regenerate the figure with `python3 -m src.models.gw_decision_flow`.
