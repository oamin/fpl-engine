# Half-season chip plan

Locked with Gemini on 2026-10-04 ([half plan](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). Phase 1 is `plan_half` in `src/live/half_plan.py`. Gemini kept that function. `score_xp` stays the published score. A double stays one fixture. The historical climb keeps an empty chip map. Phases 2 and 3 score the bench inside a transfer only when the caller names the planned Bench Boost week. The published climb does not name one. Twenty-four unit tests passed, and Gemini kept the bench term.

The layer chooses chips. It does not price players. Each deadline it looks to the end of the current half, Gameweek 19 or Gameweek 38, using the squads and the fixture list already in hand. It is solved again at the next deadline. A double that has not been announced is not in the plan.

Chip law stays in `src/rules/fpl_2026.py`. One of each chip per half. An unused first-half chip is worth nothing from Gameweek 20. Gameweek 1 cannot play Wildcard or Free Hit. One chip in a week. No Free Hit in consecutive weeks. The new module imports that wallet. It does not restate the dates or the names.

## Value of a schedule

A schedule gives each chip a week in the remaining half, or leaves it unused. Illegal schedules are those the wallet rejects.

The squad in view is the one already held. From the wildcard week onward, the squad in view is the rebuilt fifteen. The caller prices both squads. This function does not pick the fifteen.

A Free Hit week scores the Free Hit eleven only. The bench is not added, and the squad in view is back the next week.

Any other week scores that squad's eleven, plus the bench if Bench Boost is that week, plus one extra copy of the captain if Triple Captain is that week. That copy is the captain's expected points once, on top of the normal double. An unused chip adds nothing. The weeks are not discounted. The three-week discount stays inside the transfer search, which this function does not touch.

Weeks beyond the three-week horizon reuse the last priced step. They do not grow a new projection. The points are the published single-fixture scores, so a double is understated. That limit stays until the double-fixture question is opened on its own.

## What is played this week

Take the best legal schedule. If it plays nothing this week, play nothing.

Bench Boost is played when this week is the best week left in the half for it, and the bench is worth more than zero. It may be an ordinary week. Waiting for a double would expire the first-half chip before Christmas in a year with no early double.

Triple Captain is played when this week is the best week left for it, and the extra captain copy is worth more than zero.

Free Hit is played when this week is the best week left for it, and the Free Hit eleven beats the squad's own eleven by at least 12 on that week alone.

Wildcard is played when this week beats every later week as the moment to play it, and the rebuilt fifteen beats the held fifteen by at least 16 across the rest of the half. The 16 is that half-season gap. It is not required of a single week. A one-week hurdle would silence a wildcard whose gain arrives through a later Bench Boost.

If two chips tie for this week, play nothing.

The margins 12 and 16 stay the live margins. They are not refit.

## Phases

Phase 1 is `plan_half` in `src/live/half_plan.py`, with tests in `tests/test_half_plan.py`. The caller passes the week tables. The function returns the chip for this week and the schedule that matches that action. It does not replace `recommend_chip`. It does not call the squad picker, the climb, or the network. `bench_week` reads the Bench Boost week from that schedule when the week is still ahead, including this week. A used chip, or a schedule with no Bench Boost, returns nothing.

Phases 2 and 3 are the optional `bench_gw` argument on `transfer_value` and `choose_transfers` in `src/models/season_climb_ft.py`. The bench is the four players left out of the XI, scored with the same `score_xp` column as the XI. It is added on one week.

- When that week is one of the next three, only that step is added, discounted by γ to the power of the step. The other two steps stay at zero.
- When that week is later than the three, the decision-week bench is added once, discounted by γ to the power of the gap from this week to the chip.
- A week already passed, or a week missing from the horizon list, adds nothing.
- The default is no week. `run_ft_season` leaves the argument off, so the published climb still scores the bench at zero.

There is no price term and no separate sale. After the chip is used the caller stops passing the week. An equal eleven with a cheaper bench then fails the same hold that already sits on a transfer, so the bench is sold only when an eleven gain pays for it. A historical climb that fills in the chip map is still a later lock. That climb would move published season totals.

## What the plan still does not see

Transfers between this deadline and the Bench Boost week are invisible until the next solve. The wildcard fifteen is whatever the caller priced. The Gameweek 6 example is in the chip sum: the wildcard's value includes the Bench Boost it makes available before Gameweek 19, and the 16-point hurdle is the eleven only. The transfer search pays for the bench on the chip week, and the discount on a chip thirteen weeks away is small, so the strong bench is built when that week is close.
