# String agent — move accounting

Gemini ([carry review](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)) locked this slice before it was coded. The string agent still does not call `score_xp`. The official ledger `data/predictions/2026-27/string_agent_freeze.jsonl` is not written.

## What a legal move is

The draft's ins and outs must be exactly the players who joined and left the carried fifteen. A player cannot be in both lists. Captain and vice must be two different starters.

Selling price is the entry's selling price when the snapshot has one, otherwise `sell_price` from `src/rules/fpl_2026`. A missing purchase price fails the move. The bank after sales and buys must be at least 0. The old £100m squad cap is not applied to a carried squad.

Hits are `hit_cost`. Bench Boost and Triple Captain do not wipe them. Wildcard and Free Hit do. Wildcard does not add a free transfer for next week: 1 before and 5 transfers leaves 1. A Free Hit leaves the next squad, the next bank, and the purchase prices as they were, and it does not spend the free-transfer bank. Two free hits in a row are refused.

## Dry run

`python -m src.str_agent --entry 2632584 --hold`

ojaminFC, gameweek 6, no transfers, no chip. Legal. Bank stays 15. Hits 0. The free-transfer bank for the next deadline is 2. The only chip on the carry is the gameweek 1 triple captain. `--commit` is refused.
