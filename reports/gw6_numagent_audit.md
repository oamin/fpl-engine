# Gameweek 6 numagent audit

Iteration `numagent-gw6-lock`, commit `6638dce8baee02071c3f1461f065face9bd50863`. The score file `gw06/20261010T091116Z.csv` was written at 2026-10-10T09:11:16Z. The freeze was written at 2026-10-10T09:27:29Z, before the 10:00 deadline. Source hash `69da7110464578f4f546167d559bd3405bc4e225e32f1fd0bdca03a246d83c55`.

This is the code that scored the Saturday wildcard. Anytime rates still pass through the residual team budget. The later iteration `numagent-anytime-full` (`57ea53d`) applies each minutes-scaled rate in full. The lock was not re-scored on that later code.

Entry 2632584, ojaminFC. Bank 15 tenths. One free transfer. Triple Captain was already played in Gameweek 1. Chips still available: wildcard, free hit, bench boost. Squad value 1010 tenths. Played through Gameweek 5, 350 points.

## What the scorer opened

The score file stamps these hashes on every row. They match `gw06/lock_gw6.json`.

| Input | File | What it supplied | Hash |
| --- | --- | --- | --- |
| Player history | `data/cache/player_gw_2026_27.csv` at the retired cache hash | Shot shares and the buy-gate count, rows with `gw` < 6 | `ffe911b65b5019529f8a7a322ad058e013a0e6146677c35d796f075e2572f09b` |
| Historical odds | `data/cache/E0_2627.csv` | Football-data 1X2 and over/under, then the Exchange rows are appended | `8e21d4821be776332f3504f55f85b4277f326f03513b2c7501fbdedf92b8b1a2` |
| Exchange lines | `gw06/betfair_t1/gw_lines.csv` | 20 Betfair rows: 10 in Gameweek 6, 10 in Gameweek 7 | `98e5bb4281312953f19a18383d2d6691e34279618c900c736d39c22085bb1b0a` |
| Minutes | `gw06/xmi_t1.csv` | 667 rows of `player_id,gw,xmi,tag`. The score uses `xmi` | `efb2c9b0cb1be3de920b75bf0f5f5f1cace273389576971cd0b9d19ef382067d` |
| Prices and clubs | `gw06/bootstrap_20261010T082509Z_t1.json` | 667 elements, 20 clubs, current cost | `f53c43b2c822d111bca9e3ae615ceb539ff324fbf9fae9a07ded738891b01546` |
| Fixtures | `gw06/betfair_t1/fixtures.json` | 380 fixtures, 10 a week through Gameweek 38 | `477fee196153c242d686d57bb9605fedb219ebdcdb8ee09b33660fc52cd8c120` |
| Anytime | `gw06/betfair_t1/betfair_to_score.json` | 12 runners | `c5b4ae7591fe81c80e0eb4dac8f2a65b9465088fb1b843cae9c4abfb25dc051b` |
| Outrights | `gw06/betfair_t1/outrights_ranks.json` | 19 clubs | `85f2818ac767a3878618fe3cb357c2a5e86a82111f7002edbb1c683b3789805c` |
| Owned squad | `data/entry/2632584.json` | Bank, free transfers, chips already played, the Gameweek 5 fifteen | `3049c664dd1b2e517d0ef8b59c0078d485f327951d4603e1bc460e11248013c2` |

`official_20261010T082509Z_t1.csv` is 667 rows of `ep_next` for Gameweek 6, hash `de9775a6f44aa61513956e0b367331ad0e693df077566e8cc6cd214fd905c5ac`. It was stored on the freeze after the squad had been chosen. It did not choose the squad.

The string dossier, the Odds API, `data/live/gw_lines.csv`, and `data/live/xmi_gw6.csv` are not inputs to this score. Packet text is not a term in the score. The minutes sheet is.

On the Exchange sheet, Nott'm Forest v Arsenal in Gameweek 7 has no over/under, so that match uses an even total. The outright table has no Nott'm Forest row. A missing club keeps the bottom-table strength, expected rank 18.5.

The anytime file is eleven Arsenal v Leeds runners plus Liam Delap in Crystal Palace v Nott'm Forest. Saka's raw rate is 0.399. In the lock file his Gameweek 6 score is 3.32.

Minutes on the sheet: 445 rolling avg, 99 injured, 101 transferred, 21 at 50/50, 1 benched. 305 players are on 0 minutes. 73 are on 90. Haaland is on 90.

The written score file has 213 players in each of Gameweeks 6, 7, and 8. That is the pool after the buy gate, with every owned player kept.

## What the model is

The hashed source set is the rules, the points formula, the chip margins, the squad search, and the deadline scorer. Session code is outside that hash.

1. Shares. Finished rows with `gw` < 6. A player's share of goals is his shifted expected goals over his club's shifted expected goals. Assists are the same on expected assists. Defensive-contribution rate is the shifted hit rate. The deadline week itself is a stub and is not inside the share. A club with no history uses a neutral team rate of 1.40 expected goals and 1.05 expected assists.

2. Pool. A player can be bought with at least three prior appearances and expected minutes of at least 45. Every player already owned stays in the pool. The forward goal scale stays 1.

3. Match pot. Shin removes the overround on the 1X2. Expected goals in the match are 2.45 plus 1.10 times the gap between the over-2.5 and under-2.5 probabilities. Each club's lambda is that total times its attack share. Assists are 0.75 of lambda. A missing total is an even 2.5 market. A double would use the first pot. Gameweeks 6 and 7 each have their own line.

4. One-match score, `xp_on_pot`. Appearance is the chance of playing at least one minute plus the chance of playing 60, from minutes over 15 and over 30. Goals are share times lambda times the rules points: goalkeeper and defender 6, midfielder 5, forward 4. Assists are share times assist-lambda times 3. Clean sheets, defensive contributions, and goalkeeper saves use the 60-minute chance. Bonus points are 0.18 of the goal term, 0.12 of the assist term, and 0.08 of the clean-sheet term. Goals conceded and a yellow-card term are subtracted.

5. Anytime overlay, this iteration only, Gameweek 6 only. Each priced rate is scaled by minutes over 90. Unpriced teammates keep their historical share of the club's lambda. The priced rates are then scaled down into the leftover budget, and that budget is at least 0.1. Later weeks do not use the anytime file. They stay on historical shares.

6. Gameweek 8. Clubs without their own 1X2 take a pot from outright strength, shrunk halfway to a neutral pot (`OUTRIGHT_SHRINK` 0.5). Gameweek 9 repeats that Gameweek 8 row. It is the same pot again.

7. Squad. `rebuild_squad` buys a new fifteen on the Gameweek 6 `score_xp`, paid from the bank plus the rules sell prices. It is not a fresh £100.0m. The same fifteen is scored again on the Gameweek 7 step. `choose_transfers` searches one sale. The second free transfer is off. The eleven, captain, and vice come from `live_xi` on that fifteen. The captain is the highest score in the eleven, counted a second time.

8. Chips. `plan_half` sums only Gameweeks 6 and 7. A copied week adds nothing. Wildcard needs the rebuilt eleven to lead the untouched eleven by 16 across those two weeks. Free Hit needs a lead of 12 on Gameweek 6 alone. On the decision week the free-hit eleven is the same rebuild, so those two chips tie on that week. Bench Boost and Triple Captain play this week only when it is strictly the best priced week for that chip. `GW6_CHIP_COUNTS` is false. The freeze records the wildcard that was submitted. That flag stays false, so the row is a logged judgement.

## What that run stored

Wildcard. Formation 4-4-2. Mbeumo captain, Fernandes vice, Haaland starts on 5.02 from 90 minutes. Bank left 5 tenths. No hit. Fernandes, Rogers, Haaland, and Davis stay from the Gameweek 5 fifteen.

Gameweek 6 eleven with the captain counted twice: wildcard 63.14, untouched 53.02, one transfer (van Ewijk to Silva) 54.33. The wildcard leads the untouched squad by 18.06 over Gameweeks 6 and 7, which clears 16. The same rebuild leads that one-transfer path by 13.88, which is under 16. The 138.38 figure is the Gameweek 6 rebuilt eleven, the Gameweek 7 rebuilt eleven, and the Gameweek 7 rebuilt bench. Gameweek 8 is outside that sum.

The schedule also names Bench Boost in Gameweek 7, because that priced bench is the larger of the two. It is read again when Gameweek 7 has its own line.

On `ep_next`, the same three elevens are 56.0, 49.5, and 51.5. Those are the official column on the already chosen lineups.
