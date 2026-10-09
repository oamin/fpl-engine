# Gameweeks 1–5 under the current climber

The recorded fresh climb stays 280. The recorded climb from ojaminFC's Gameweek 1 fifteen stays 333. Their own total stays 350. This note is a causal replay of today's free-transfer climber on `score_xp`, with an empty chip map. It does not replace those figures, and it is not the Gameweek 6 lock.

Running today's climber from a fresh buy pool yields **315**. Starting from the same Gameweek 1 fifteen yields **339**, net of one 4-point hit. Gemini accepted this reading ([gw15 rerun](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)). A same-week minutes oracle was not run.

## Fresh squad

The 6 October climb in `reports/own_squad_gw15.md` is 280: Gameweeks 40, 89, 48, 62, and 41. It keeps that fifteen through Gameweek 2, sells O'Reilly and Hincapie for Van Hecke and Thiaw in Gameweek 3, and then holds. Hits are 0.

Today's climber scores 315, also with 0 hits: Gameweeks 38, 96, 61, 72, and 48. The Gameweek 1 fifteen is a different squad, so the 35 points are not a forecast laid on top of the 280.

| GW | Recorded | Now | Formation now | Captain now | Transfers now |
|---:|---:|---:|---|---|---|
| 1 | 40 | 38 | 1-4-5-1 | B.Fernandes | solved from the buy pool |
| 2 | 89 | 96 | 1-4-4-2 | B.Fernandes | hold |
| 3 | 48 | 61 | 1-4-5-1 | Guéhi | in Schade, Van Hecke; out Enzo, Konsa |
| 4 | 62 | 72 | 1-5-3-2 | Khalaili | in Khalaili; out Senesi |
| 5 | 41 | 48 | 1-5-3-2 | B.Fernandes | hold |
| Total | 280 | 315 |  |  | 0 hits |

The recorded fifteen is Sánchez, Gabriel, Guéhi, Hincapie, O'Reilly, Anderson, B.Fernandes, Mbeumo, Ndiaye, Schade, Thiago, Verbruggen, Colwill, Calvert-Lewin, and Šeško.

The fifteen solved now is Donnarumma, Tarkowski, Senesi, Enzo, E.Le Fée, Welbeck, Darlow, Konsa, Gabriel, Guéhi, B.Fernandes, Mbeumo, Anderson, Thiago, and Calvert-Lewin. The Gameweek 1 scoring eleven is Donnarumma, Gabriel, Guéhi, Tarkowski, Senesi, B.Fernandes, Mbeumo, Enzo, Anderson, E.Le Fée, and Thiago.

## From the Gameweek 1 fifteen

`reports/stage_42_gw15_horizon.md` is the published path from that fifteen: the opening horizon plus the early score, **333**, with 0 hits and Sangaré held. Weeks are 72, 98, 47, 80, and 36. The freeze arm on the same fifteen is 336 with 8 hits. The horizon without the early score is 366. ojaminFC's own transfers and the Gameweek 1 Triple Captain scored 350: weeks 62, 108, 53, 79, and 48.

Today's climber, given that same fifteen and then left to transfer, scores **339**. Gameweek 1 is the same 72. The path then takes one hit.

| GW | Published path | Now | Hits now | Captain now | Transfers now |
|---:|---:|---:|---:|---|---|
| 1 | 72 | 72 | 0 | B.Fernandes | none; the fifteen is already owned |
| 2 | 98 | 102 | −4 | B.Fernandes | in Darlow, Szoboszlai; out Martinez, Ødegaard |
| 3 | 47 | 44 | 0 | Haaland | in M.Bizot; out Darlow |
| 4 | 80 | 84 | 0 | João Pedro | in Kinsky; out M.Bizot |
| 5 | 36 | 37 | 0 | Haaland | hold |
| Total | 333 | 339 | −4 |  |  |

Gameweek 2's eleven, captain doubled, is 106 before the hit. One extra transfer costs 4, and 106 − 4 = 102. The five weeks that are added are 72, 102, 44, 84, and 37. That sum is 339. Adding the 4 back, to 343, is not the score.

The published 333 took no hit and held Sangaré. This replay sells Martinez and Ødegaard in Gameweek 2. It is not the 333 path plus 6.

The opening fifteen is Ødegaard, João Pedro, Martinez, Guéhi, Cherki, Haaland, Shaw, B.Fernandes, Gibbs-White, Scarlett, Forster, Calafiori, M.Sangaré, van Ewijk, and Davis. Gameweek 1 is 1-3-5-2, with one automatic substitute.

## What moved, and what did not

The two gaps, +35 on a fresh squad and +6 from the opening fifteen, were not split apart. Four things differ from the climber that produced 280 and 333, and this replay did not turn them off one at a time:

- The pool is the sheet, including rows with 0 minutes. The 280 used the played-only pool. A buy still needs expected minutes of at least 45.
- The formation list is the official eight, including 5-2-3. The 280 used a shorter list. This fresh path played 5-3-2 in Gameweeks 4 and 5.
- A goalkeeper goal is worth 6, from `src/rules/fpl_2026.py`.
- Later weeks in the three-week value use `forecast_xp`: that fixture's opening pot, with a shrunk outright when the pot is missing. The decision week still uses `score_xp`. The names `attach_opening_horizon` and `attach_forecast_xp` are the same function. The October horizon that produced 333 was the opening line of that date.

Expected minutes stay the lagged prior. A 0-minute row is still dropped before that prior updates. Setting this week's expected minutes to the minutes that were played is lookahead, and it is not a figure in this note. Zeroing Sánchez from Gameweek 3, in the earlier news-tag trial, left the 280 where it was, because Verbruggen was already the automatic substitute.

The replay also needed a join fix. Player logs now carry `value`, and the sheet join was creating `value_x` and `value_y`, so the price column disappeared. The log price is kept. A gap takes the sheet price. That repair does not change a price both copies already share.

`reports/own_squad_gw15.md` and `reports/stage_42_gw15_horizon.md` are unchanged. No row was added to `experiments/matrix.json`. No Odds API call was made. The Gameweek 6 numeric lock remains the Saturday `ep_next` freeze.
