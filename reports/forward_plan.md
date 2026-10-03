# Forward plan

An outside review treated the one-gameweek note as the whole forecast. Some of that review describes the note. The formula in `compute_xp` is wider. This plan keeps the points that match the code, drops the ones that do not, and locks an order with Gemini.

The gap already measured is actual points. From ojaminFC's Gameweek 1 fifteen, the past-only published path scored 313 against their 350 over Gameweeks 1–5. It is not a gap between two predictions.

## What the score already contains

`score_xp` is the sum of appearance, goals, assists, clean sheets, defensive contributions, goalkeeper saves, a small bonus proxy, minus goals conceded and a yellow-card proxy. Goals and assists are shot share times the team rate. The other terms are gated by the minutes prior. A separate start probability is not in the formula. Red cards, own goals, and missed penalties are not separate terms. Defensive contributions are added for every season in the cache, including seasons whose official points did not have that award.

The first job is to measure those pieces against actual points. It is not to rebuild the attacking term.

## What the review got right

The three-week value reuses this week's score. It does not rebuild the rate for the next opponent. A future week in which the club has no fixture is already set to 0 for that week only. A blank this week is different: today's score is set to 0, and that 0 is what the later weeks copy, even when the club plays. Doubles are scored once. `fixture_counts` is not passed in.

The opening 15 maximises the sum of all fifteen scores. Every later week maximises the XI only, and the bench is worth 0 in that value. Those are different objectives. Replacing the opener with an XI-only sum, and leaving the bench worth nothing, was rejected. Under a £100m cap that buy fills the bench with the cheapest bodies, and the season then spends free transfers repairing it. Autosubs are the reason that matters: in 2023/24 they covered 65 of 86 intended blanks, and in 2024/25 they covered 38 of 42.

The search only tries same-position swaps, then a second swap from the best six, then a third only if two is ahead. A cross-position pair exists in the code and is switched off.

5-2-3 is a legal shape and is absent from the published list. Putting it into that list would move every published total. A count, with the list left as it is, does not.

The 45-minute buy rule is a gate, not a minutes forecast. Historical injury, doubt, and suspension are not on these sheets in a form known before the deadline. A live season can store them only as a snapshot taken before that deadline.

## What stays as it is

The hold margin of 1.25 and the penalty of 1.0 are not retuned. Penalty 0 was ahead on one season and behind on the next. Penalty 2.0 lost all three earlier seasons and left more blank starters. The discount of 0.9 stays. Chips stay off in the historical climb. The empty week in 2022/23 stays skipped, and the free-transfer count does not move, because that week was not played under its own number. The three-player club cap is already enforced on every candidate squad.

## Order

**0A. Calibration table, no formula change.** On 2025/26, Gameweeks 5–38, compare `score_xp` with actual points by position and by component: appearance, goals, assists, clean sheets, defensive contributions, saves, the bonus proxy, goals conceded, and the card proxy. A component whose mean error is inside 0.10, and whose removal moves rank correlation by less than 0.01, is left alone.

**0B. One change to the three-week value, after 0A is reviewed.** Later weeks keep the decision-week share and minutes prior, and take that fixture's own pre-deadline scoring rate and clean-sheet probability from odds already on disk. A week with no fixture is 0 for that week only. A blank this week must not zero a later week that has a fixture. No double multiplier. No new odds call. γ, the hold margin, the penalty, and `score_xp` stay put. The screen is the 2023/24 free-transfer climb, Gameweeks 5–38, because that season contains the Gameweek 29 blank. The fast XI cannot see this change.

Park 0B if the season total is not higher, or if the total on weeks where all twenty clubs play is not higher. A gain that exists only on the blank weeks is not a pass. A pass on 2023/24 does not replace the published value. The other seasons are the next gate, under the same rule.

Reading a future feature row is not allowed. That row's minutes prior includes matches after the deadline.

**1. Defensive contributions on earlier seasons, only if 0A shows they are phantom points.** Set that term to 0 for seasons before 2025/26. No new parameter.

**1. The opener is not switched to an XI-only sum in this batch.** It stays the sum of fifteen until a later rule gives the bench an explicit playing floor. That floor is not designed here.

**2. Counts, not replacements.** How often 5-2-3 would be the best shape, without changing the published list. One season with the cross-position two-transfer switch on, against the current beam. Availability and the minutes gate wait for a snapshot taken before each deadline.

## Not in this batch

A new attacking model. A refit of the hold margin or the transfer penalty. A chip search. An autosub term inside the squad value. An injury flag built from minutes or from the end-of-season status file. A double-gameweek multiplier. That last one is real, and it waits until 0B has a result, so a double and a new rate are not changed in the same run.
