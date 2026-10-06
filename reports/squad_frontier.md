# Squad frontier

Counted after the lock in `reports/squad_frontier_plan.md`. The gap is model `xi_xp` minus human `xi_xp`, divided by the priced steps that chip uses. A wildcard sums the window. A free hit uses the decision week. Realised points are the fifteen's points that week, and they do not choose the call. The score is not changed. The force-in of extra human players was not run.

Chip weeks: 10. The reference manager is absent. Bench Boost and Triple Captain are absent.

## Reachability

A reachable week is a legal fifteen from the model's pre-chip bank and sell prices, with every player in that week's pool and a decision-week score already on the row. Money is a squad that passes shape and the club cap and costs more than that budget. Pool is a player missing from the pool or missing a score. Rules is squad shape, the club cap, or a fifteen that cannot form an eleven. A rules mismatch is the human's own budget rejecting the fifteen.

| Chip | Weeks | Reachable | Money | Pool | Rules | Mismatch |
|---|---:|---:|---:|---:|---:|---:|
| wildcard | 5 | 0 | 1 | 4 | 0 | 0 |
| free_hit | 5 | 0 | 0 | 5 | 0 | 0 |

## The call

The call uses reachable weeks only. Each of those weeks has equal weight. Near is a per-week gap of at most 1.0. Far is above 4.0.

| Chip | Reachable | Near | Middle | Far | Near share | Far share | Mean gap | Call |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| wildcard | 0 | 0 | 0 | 0 |  |  |  | none |
| free_hit | 0 | 0 | 0 | 0 |  |  |  | none |

wildcard: This chip has no reachable week, so it has no call.

free_hit: This chip has no reachable week, so it has no call.

The human's own budget could not be priced on 4 weeks. Those weeks are not a rules mismatch.

A gap on a week that is not reachable is printed here and stays out of the call.

Mark Brookes, Gameweek 5 wildcard: per-week gap 4.33, status unreachable_money, shortfall 0.3. Human spend by position, in £m, is goalkeeper 9.0, defence 25.2, midfield 38.9, forward 27.2. The rebuild is goalkeeper 10.0, defence 25.9, midfield 34.5, forward 29.4.

Each pool week has at least one player with no row on the published frame that week. The frame keeps a player once he has three prior appearances. No score is filled in for the weeks below.

| Manager | GW | Chip | Player | Minutes |
|---|---:|---|---|---:|
| elevenify.com | 3 | wildcard | O'Shea | 90 |
| elevenify.com | 3 | wildcard | Slater | 90 |
| Cameron Scott | 3 | free_hit | O'Reilly | 0 |
| Cameron Scott | 3 | free_hit | Mendy | 70 |
| Ashley Marsh | 3 | wildcard | M.Sangaré | 45 |
| Ashley Marsh | 3 | wildcard | Bentley | 0 |
| Mark Brookes | 3 | free_hit | Egan | 90 |
| Mark Brookes | 3 | free_hit | Ajayi | 90 |
| Will Morrison | 3 | free_hit | Dubravka | 0 |
| Will Morrison | 3 | free_hit | Egan | 90 |
| Thomas O'Brien | 3 | wildcard | M.Sangaré | 45 |
| Thomas O'Brien | 3 | wildcard | Bentley | 0 |
| Thomas O'Brien | 3 | wildcard | Muharemović | 90 |
| Filip Stripaj | 5 | free_hit | Dubravka | 0 |
| Filip Stripaj | 5 | free_hit | van Ewijk | 0 |
| daniel bowes | 5 | free_hit | Bentley | 0 |
| daniel bowes | 5 | free_hit | Kipré | 0 |
| daniel bowes | 5 | free_hit | Hughes | 0 |
| daniel bowes | 5 | free_hit | Fletcher | 0 |
| Sion Jones | 3 | wildcard | Egan | 90 |

## Weeks

Overlap is how many of the human fifteen are in the rebuild. Shortfall is the model's budget deficit, in £m. Points are descriptive.

| Manager | GW | Chip | Status | Gap | Overlap | Buy pool | Shortfall | Human points | Model points |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Cameron Scott | 3 | free_hit | pool |  | 4 | 13 | 0.0 | 52.00 | 53.00 |
| Filip Stripaj | 5 | free_hit | pool |  | 5 | 13 | 0.2 | 88.00 | 58.00 |
| Mark Brookes | 3 | free_hit | pool |  | 4 | 13 | 0.6 | 70.00 | 53.00 |
| Will Morrison | 3 | free_hit | pool |  | 3 | 13 | 0.0 | 57.00 | 53.00 |
| daniel bowes | 5 | free_hit | pool |  | 3 | 11 | 0.0 | 44.00 | 58.00 |
| Ashley Marsh | 3 | wildcard | pool |  | 2 | 13 | 0.1 | 44.00 | 53.00 |
| Mark Brookes | 5 | wildcard | unreachable_money | 4.33 | 2 | 15 | 0.3 | 64.00 | 48.00 |
| Sion Jones | 3 | wildcard | pool |  | 3 | 14 | 0.0 | 60.00 | 53.00 |
| Thomas O'Brien | 3 | wildcard | pool |  | 3 | 12 | 0.3 | 40.00 | 53.00 |
| elevenify.com | 3 | wildcard | pool |  | 3 | 13 | 0.5 | 47.00 | 58.00 |

Gemini kept the count ([squad frontier](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)).
