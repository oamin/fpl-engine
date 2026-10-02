# Live modeller

The historical climber is unchanged. This path plans the deadline that is still ahead.

## Inputs

- FPL bootstrap and fixtures, from the free API, cached under `data/live/` (not committed).
- Minutes: a CSV with `player_id`, `gw`, `xmi`. Gemini supplies it. Anyone omitted is 0 minutes. The file replaces only the minutes prior. Goal, assist, and clean-sheet shares stay on the existing engine.
- Odds: a CSV snapshot already on disk. The Odds API is refused.

`python -m src.live` refreshes the slate and stops. On 2 Oct 2026 that slate was GW6, deadline 2026-10-10 10:00 UTC, 10 fixtures, no blank and no double.

## Mechanics

Captain is the highest score in the XI. The vice-captain is second. The bench is the existing bench order. Formations are the official list, including 5-2-3. The published climb still omits 5-2-3.

Chips use expected points for the current half. A chip is played this week only when this week is the best legal slot:

- Free Hit on a blank, and only when it beats the no-chip XI by 12 expected points.
- Wildcard when the rebuild beats the constrained path by 16.
- Bench Boost on the double with the largest bench score, if that double is this week.
- Triple Captain on the double with the largest single-player score, if that double is this week. The gain versus a normal captain is one extra copy of that player.
- A tie between two chips plays nothing.
- Wildcard and Free Hit stay illegal in GW1. Free Hit cannot be played in consecutive weeks. Bench Boost in GW1 is legal.

The score itself is still an input. This slice does not build λ, and it does not pick a squad until the minutes file is present.
