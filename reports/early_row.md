# Early row

Counted after the lock in `reports/early_row_plan.md`. A player with one or two prior appearances is scored with that early number, capped at 6, including when the model does not own him. The number evaluates the human fifteen. It does not enter the buy pool. A player with 0 minutes and no score row stays unscored, and that week stays out of the call. Mark Brookes in Gameweek 5 is the money week already counted, and it is not in this table. The chip sum is still the undiscounted `xi_xp`. This reading does not add a decay.

A later priced week repeats the deadline score when the shot share is missing. That is the existing step for a player with no share.

Pool weeks: 9. The reference manager is absent.

## Reachability

The bars are unchanged. Near is a per-week gap of at most 1.0. Far is above 4.0. The call needs half the reachable weeks of that chip.

| Chip | Weeks | Reachable | Money | Pool | Rules | Mismatch |
|---|---:|---:|---:|---:|---:|---:|
| wildcard | 4 | 1 | 0 | 2 | 0 | 1 |
| free_hit | 5 | 0 | 1 | 4 | 0 | 0 |

## The call

| Chip | Reachable | Near | Middle | Far | Near share | Far share | Mean gap | Call |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| wildcard | 1 | 0 | 0 | 1 | 0.00 | 1.00 | 4.52 | far |
| free_hit | 0 | 0 | 0 | 0 |  |  |  | none |

wildcard: The human portfolio is expensive on the model's objective. The realised points sit beside this call. They are not the reason for it.

free_hit: This chip has no reachable week, so it has no call.

The wildcard call is one week. Sion Jones in Gameweek 3 is the reachable week, and the per-week gap is 4.52. Three of his fifteen are in the rebuild. Fourteen were already in the buy pool. Egan is the early score, and that row is not buyable. His spend, in £m, is goalkeeper 9.5, defence 24.0, midfield 35.9, and forward 30.6. The rebuild is 9.0, 26.5, 35.0, and 29.5. His fifteen scored 60 and the rebuild scored 53. Those points sit beside the call.

elevenify.com is fully scored and stays out. His own budget rejects the fifteen. The per-week gap is 2.84, which is the middle band. He scored 47 and the rebuild scored 58. Mark Brookes in Gameweek 3 is fully scored and stays out. The fifteen costs £0.6m more than the model's pre-chip budget. The per-week gap is 4.66. He scored 70 and the rebuild scored 53.

The other six weeks each still contain a player who played 0 minutes. All ten of those slots are tagged replaced: a teammate at the same club and position played at least 60 minutes. None is benched, none is a club with no game, and none is unresolved.

A gap on a week that is not reachable is printed here and stays out of the call.

elevenify.com, Gameweek 3 wildcard: per-week gap 2.84, status rules_mismatch, shortfall 0.5.

Mark Brookes, Gameweek 3 free_hit: per-week gap 4.66, status unreachable_money, shortfall 0.6.

## Weeks

Early players are the ones scored from the capped table. Blank tags are the 0-minute slots. Points are descriptive.

| Manager | GW | Chip | Status | Gap | Early players | Blanks | Human points | Model points |
|---|---:|---|---|---:|---|---|---:|---:|
| Cameron Scott | 3 | free_hit | pool |  | Mendy | O'Reilly replaced | 52.00 | 53.00 |
| Filip Stripaj | 5 | free_hit | pool |  |  | Dubravka replaced; van Ewijk replaced | 88.00 | 58.00 |
| Mark Brookes | 3 | free_hit | unreachable_money | 4.66 | Egan; Ajayi |  | 70.00 | 53.00 |
| Will Morrison | 3 | free_hit | pool |  | Egan | Dubravka replaced | 57.00 | 53.00 |
| daniel bowes | 5 | free_hit | pool |  |  | Bentley replaced; Kipré replaced; Hughes replaced; Fletcher replaced | 44.00 | 58.00 |
| Ashley Marsh | 3 | wildcard | pool |  | M.Sangaré | Bentley replaced | 44.00 | 53.00 |
| Sion Jones | 3 | wildcard | reachable | 4.52 | Egan |  | 60.00 | 53.00 |
| Thomas O'Brien | 3 | wildcard | pool |  | M.Sangaré; Muharemović | Bentley replaced | 40.00 | 53.00 |
| elevenify.com | 3 | wildcard | rules_mismatch | 2.84 | O'Shea; Slater |  | 47.00 | 58.00 |
