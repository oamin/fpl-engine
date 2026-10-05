# What the published rules are

This file states the frozen engine. It does not add a result. Gameweeks 1–5 of 2026/27 are a diagnostic trace. They are not a sample for changing a number.

The active research is the cohort chain that runs from the Gameweek 1 fifteens. `main` is behind that chain. A number in a draft report belongs to the branch named in that report.

## Frozen rules

- Player score: `score_xp`, one fixture, from the opening 1X2 and the 2.5 line.
- Hold margin: 1.25. Switch penalty: 1.0.
- Free Hit needs 12 expected points over the held eleven. Wildcard needs 16 on the priced weeks only, and those weeks stop at Gameweek 7.
- The eleven is the legal shape with the highest sum of `score_xp`. The shapes considered are 3-4-3, 3-5-2, 4-4-2, 4-3-3, 4-5-1, 5-3-2, and 5-4-1.
- Chips are chosen from expected points. Realised points are the score afterwards, not the input.

## What this window has already said

The model was ahead of none of the 14 carried fifteens. The gap is about the players in the squad and about who starts. Switching the eleven to five midfielders, on the players the model already owns, returned about a point and a half a week and did not clear the bar. Three forwards scored less. The formation tag of −163 is not a pile of points a different shape would have collected. `score_xp` stays frozen.
