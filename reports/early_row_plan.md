# Early row

A plan, locked on 2026-10-06 before any new squad gap is read. Gemini kept it ([early row](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)). Nothing here changes `score_xp`, the hold of 1.25, the wildcard margin of 16, the free-hit margin of 12, or the formation list. The climb still buys a player only after three appearances. ojaminFC stays out. Entry 1078627 stays out. No Odds API call.

## What the three requests meet

A week with 0 minutes is removed before a score is built. A week with fewer than three prior appearances is removed after it. An early score, capped at 6, exists for one or two prior appearances, and the chip pool attaches it only for a player the model already owns.

Ten of the twenty missing slots played in Gameweek 3 and already have that early score: O'Shea 3.15, Slater 3.56, Mendy 4.32, M.Sangaré 5.92, Egan 4.24, Ajayi 4.10, and Muharemović 3.49. None of those ids has a linked 2025/26 row. Their first published row is Gameweek 4. Sangaré is in two squads and Egan is in three. Three of the nine pool weeks contain only these players: elevenify.com, Mark Brookes, and Sion Jones, all in Gameweek 3. The other six each contain a player who played 0 minutes and has no row.

Opening the buy gate for one or two appearances has already been run. The four template climbs were 1606 against 1735, 1960 against 2105, 2019 against 2086, and 2018 against 1988. The pool was 7603 against 7948. One season was ahead. Thirty-seven players were bought, the mean decision score was 5.50, and 22 of them scored 2 or fewer in the week they arrived. That screen stays rejected. It is not run again.

The sheets record minutes. They do not record a medical reason. A live news tag was already played through these five weeks and the climb stayed 280. A tag built from who played is known after the week. It can inform the next deadline. It is not a feature for the deadline just gone.

## Reading D

The population is the nine pool weeks from the squad frontier. Mark Brookes in Gameweek 5 stays the money week already counted, and it is not reopened.

A missing player who has an early score that week is scored with it, capped at 6, even though the model does not own him. The number is used to evaluate the human fifteen. It does not enter the model's buy pool, and it does not make him buyable.

A player with 0 minutes and no score row stays unscored. A week that still contains one stays a pool week and stays out of the call. No zero is written in, and last week's score is not carried onto a player the model does not own.

A week that becomes fully scored takes the same reachability test as the frontier. The model's pre-chip bank and sell prices, the club cap, and squad shape decide whether it enters the call. The bars stay a per-week gap of at most 1.0 for near, and above 4.0 for far. The call needs half the reachable weeks of that chip. The two chips stay separate. Realised points stay beside the call.

Each 0-minute slot is tagged beside the call. `replaced` means a teammate at the same club and position played at least 60 minutes. `benched` means the club played and nobody at that position did. `no_fixture` means the club had no game. The tag does not move a score, a minutes mean, reachability, or the call.

## What this batch does not switch on

Last season stays on a player who remained at his club. Dropping a personal rate because the club changed, or because the id has no linked season, is a different score. It waits for its own fast XI screen.

The blank does not yet enter the three-game minutes mean. That rule was amended earlier and is still off: every blank would enter the mean, a goalkeeper would be zeroed only after two consecutive `replaced` weeks, and an outfielder would not be hard-zeroed. The fast XI cannot see a player who did not play, so that screen is a transfer climb. No bar for it is set here, and the climb is not run.

The counts for Reading D are not in this note.
