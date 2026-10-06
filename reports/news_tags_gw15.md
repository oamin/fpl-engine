# News tags, Gameweeks 1–5

A tag is used only from a note dated before that gameweek's deadline. The current FPL status is not treated as the status at an earlier deadline. Minutes then follow the tag. The published climb was not run.

A firm starter keeps his old minutes, and the result stays between 65 and 90. A ruled-out player is 0. A doubtful player keeps the chance times his old minutes. A transfer is 0 at the old club. A benched goalkeeper is 0. A benched outfielder is 15. An unclear note leaves the minutes unchanged.

Deadlines: GW1 2026-08-21T17:30:00Z, GW2 2026-08-28T17:30:00Z, GW3 2026-09-04T17:30:00Z, GW4 2026-09-12T12:30:00Z, GW5 2026-09-18T17:30:00Z.

30 players share the second 2026-07-23T12:01:23. Those rows are not used, because one shared second is not a time a story was published.

## Sánchez

Gameweek 2's deadline is Friday 28 Aug 17:30 UTC. The Athletic and Guardian pieces that day say a deal for Martínez is agreed and he is due to sign. They do not say Sánchez is dropped for Sunday. Sky's report that the signing was announced hours before the Brighton match is Sunday 30 Aug, after the deadline. The Como loan is 1 Sep on the BBC and 2 Sep on the FPL line, both before the Gameweek 3 deadline.

| GW | Tag | Source | Tagged minutes | Old minutes | Played | Note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | none | none | unchanged |  | 90 |  |
| 2 | none | pending | unchanged | 90 | 0 | prose packet not classified |
| 3 | transferred | fpl | 0 | 90 | 0 | Has joined Como on loan for the rest of the season |
| 4 | transferred | fpl | 0 | 90 | 0 | Has joined Como on loan for the rest of the season |
| 5 | transferred | fpl | 0 | 90 | 0 | Has joined Como on loan for the rest of the season |

## Martínez

The same dates apply. He is not tagged as the Chelsea starter until a document dated before the deadline says so.

| GW | Tag | Source | Tagged minutes | Old minutes | Played | Note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2 | none | pending | unchanged |  | 90 | prose packet not classified |
| 3 | none | pending | unchanged | 90 | 90 | prose packet not classified |
| 4 | none | pending | unchanged | 90 | 90 | prose packet not classified |
| 5 | none | pending | unchanged | 90 | 90 | prose packet not classified |

## The model's squad

The fifteen are the published climber's own squad. A row with no pre-deadline note keeps the minutes the score already uses. Played minutes are the result after the deadline. They are not an input.

No other player in that fifteen has a pre-deadline note that changes his minutes.

## Dated FPL lines

Each count is a player whose own `news_added` is before that deadline and is not the shared stamp.

| GW | Transferred, 0 | Ruled out, 0 | Doubtful, scaled | Bench, 0 | Bench, 15 | Unclear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41 | 19 | 0 | 0 | 0 | 0 |
| 2 | 48 | 27 | 1 | 0 | 0 | 0 |
| 3 | 96 | 35 | 2 | 1 | 0 | 0 |
| 4 | 100 | 44 | 3 | 1 | 0 | 0 |
| 5 | 101 | 59 | 6 | 1 | 0 | 0 |

Doubtful players keep a share of their old minutes. They are not zeroed. A player with no earlier appearance uses 90 as the full match, then the chance.

| GW | Player | Chance line | Old minutes | Tagged minutes | Played |
| --- | --- | --- | ---: | ---: | ---: |
| 2 | Palestra | Thigh injury - 50% chance of playing | 90, no appearance | 45 | 0 |
| 3 | Caicedo | Calf injury - 50% chance of playing | 13 | 6.5 | 0 |
| 3 | Palestra | Thigh injury - 50% chance of playing | 90, no appearance | 45 | 0 |
| 4 | Caicedo | Calf injury - 50% chance of playing | 13 | 6.5 | 0 |
| 4 | Jaouen | Ankle injury - 50% chance of playing | 90, no appearance | 45 | 0 |
| 4 | Palestra | Thigh injury - 50% chance of playing | 90, no appearance | 45 | 0 |
| 5 | Caicedo | Calf injury - 50% chance of playing | 13 | 6.5 | 0 |
| 5 | James | Hamstring injury - 75% chance of playing | 67.33 | 50.5 | 0 |
| 5 | Jaouen | Ankle injury - 50% chance of playing | 90, no appearance | 45 | 0 |
| 5 | João Pedro | Knee injury - 75% chance of playing | 90 | 67.5 | 0 |
| 5 | Palestra | Thigh injury - 50% chance of playing | 90, no appearance | 45 | 0 |
| 5 | Targett | Ankle injury - 75% chance of playing | 5 | 3.75 | 0 |

## The prose packets

The prose completion has not been applied. FPL lines above stand on their own timestamps.

Cases in the prompt:

- player_id 28 | Martinez | GKP | gw 2 | deadline 2026-08-28T17:30:00Z
- player_id 140 | Sánchez | GKP | gw 2 | deadline 2026-08-28T17:30:00Z
- player_id 28 | Martinez | GKP | gw 3 | deadline 2026-09-04T17:30:00Z
- player_id 140 | Sánchez | GKP | gw 3 | deadline 2026-09-04T17:30:00Z
- player_id 28 | Martinez | GKP | gw 4 | deadline 2026-09-12T12:30:00Z
- player_id 140 | Sánchez | GKP | gw 4 | deadline 2026-09-12T12:30:00Z
- player_id 28 | Martinez | GKP | gw 5 | deadline 2026-09-18T17:30:00Z
- player_id 140 | Sánchez | GKP | gw 5 | deadline 2026-09-18T17:30:00Z

Sheet: `/workspace/data/processed/news_tags_gw15.csv`.
