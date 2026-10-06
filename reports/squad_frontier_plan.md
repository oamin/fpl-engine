# Squad frontier

A plan, locked on 2026-10-06 before any squad gap is read. Gemini kept it ([squad frontier](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)). Nothing here changes `score_xp`, the hold of 1.25, the wildcard margin of 16, the free-hit margin of 12, or the formation list. ojaminFC stays out of every mean. Entry 1078627 stays out. No Odds API call. Realised points do not choose a squad, a weight, a forced player, or the call.

## What stays frozen

The decision-margin count stands. Club or price is −163 of the −229 the rebuild left out. The 24 legal swaps are −66, split −31, −29, and −6, with shares 0.47, 0.44, and 0.09. The ranking call is inconclusive. The winner's-curse claim is parked. The top-band gaps on the four earlier seasons are +0.35, −0.10, +0.06, and +0.16.

`score_xp` stays the published score. Formation rules stay. The hold stays 1.25. Ownership and flow stay parked. A generic ranking correction, a winner's-curse correction, and a residual or role-change term are not built. The story that humans pick the slightly lower-ranked player is not a next feature. The *k*-player force-in waits until this reading is counted and reviewed.

## The arithmetic

Chip signings are −329, from 85 players. The rebuild held 19 of them, −100. It left 64 out, −229. Those 64 are club or price −163 and legal swaps −66. The −100 belongs with the −329. Adding −163, −100, and −66 and labelling the sum −229 is the wrong total.

## What the −163 measures

`place_signing` tests a one-for-one swap of the human signing for the lowest `score_xp` at that position inside the fifteen the rebuild already chose. Club means that swap would break the club cap. Price means the market price exceeds the rebuilt bank plus the sell of that one player. A block on that swap leaves open whether a different legal fifteen could hold the player. `rebuild_squad` spends the bank plus the sell prices of the pre-chip fifteen and picks a new fifteen. It does not start from a fresh £100.0m.

The historical budget climb in `reports/stage_19_season_climb_ft.md` is a different experiment. It compares a free £100m squad with a budgeted ridge on gameweeks 5–38, chips off. It does not answer this reading.

## Reading C

The population is one row per wildcard or free-hit week among the same 14 carried managers. Bench Boost and Triple Captain stay out. The reference manager stays out. The model may have played no chip that week. `one_week` still stores the rebuild, and that fifteen is the model's portfolio.

Both portfolios are scored with `squad_outlook` on the pre-deadline step scores the carry already uses. `xi_xp` is the picked eleven plus one captain copy, the maximum score in that eleven. The same function runs on both fifteens, so the human's named eleven stays out of the gap. The captain copy sits inside `xi_xp` and is not added again.

A wildcard sums `xi_xp` over the window `price_horizon` already uses: this week, plus up to two later weeks that have clubs, and nothing after Gameweek 7. The gap is the model's sum minus the human's sum. A positive gap means the model's portfolio is ahead on its own objective. The held squad is the same on both sides, so the difference matches the difference in `wc_sum`. This reading reports the `xi_xp` difference. It leaves `wc_sum` as the lead already compared with 16, and it does not compare this gap with 16.

A free hit uses the decision week's `xi_xp` only. Later horizon weeks are the squad the free hit returns to, and they stay out of this objective. The gap is the model's week minus the human's week.

The two chips are separate calls. They are not averaged. A window that stops early, because Gameweek 7 is the last priced week or because a blank is skipped, stays in the reading. The per-week gap is the raw gap divided by the number of priced steps actually used. A short window is not padded with zeros and is not dropped. Two identical fifteens have a gap of 0 and count as near.

## Reachability

This is decided before a week enters the call. The human's post-chip fifteen is costed from the model's pre-chip state. The budget is the model's bank plus the sell prices of the model's fifteen. A player the model already owns is costed at that sell price. Anyone else is costed at the market price. Club cap and squad shape come from `src/rules/fpl_2026.py`.

A reachable fifteen is a legal chip squad from that state. It enters the call. The gap is the objective sacrifice.

A fifteen that passes squad shape and the club cap, and whose spend exceeds that budget, is unreachable on money. The shortfall is reported in tenths of a million. The week stays out of the near/far call. It is the model-induced money constraint.

A fifteen that breaks squad shape or the club cap, or that contains a player absent from that week's pool, is a rules or pool row. A missing score is not filled with zero. The week stays out of the call, and it is not an xP gap. The human's own bank and purchase prices are a separate note. A fifteen the rules reject on that accounting is a rules mismatch, and it still stays out of the call. A fifteen that cannot form a legal starting eleven under the formation list is the same kind of row.

## The bars

Each reachable week has equal weight. Realised points do not weight a week.

A week is near when its per-week gap is at most 1.0. A week is far when its per-week gap is above 4.0. Anything between is middle. The figure 1.0 is the weekly scale named for a close portfolio, and it sits inside the existing hold of 1.25. It is not a new hold. The figure 4.0 is one transfer hit. The wildcard hurdle of 16 is not reused.

The call for one chip, over its reachable weeks only, is near when the share of weeks at or below 1.0 is at least 0.5. It is far when the share of weeks above 4.0 is at least 0.5. Anything else is middle. A chip with no reachable week has no call. The mean per-week gap is printed beside the call and does not make it.

A near call means the human portfolio sits close to the model's own objective, so the next suspicion is that objective or the state the rebuild started from. A far call means the human portfolio is expensive on the model's objective. Realised points may be printed beside that call. They are not the reason for it. A middle call leaves the question open. No constant moves in any of the three.

## Beside the call

These columns are descriptive. Squad price, spend by position, the maximum club count, how many of the fifteen are in the buy pool, how many are in the model rebuild, and the realised points of each fifteen.

## After this reading

The counts come next. They are not in this note. The force-in of 1, 2, 3, and further human players is a later search. It is not opened by locking this reading.
