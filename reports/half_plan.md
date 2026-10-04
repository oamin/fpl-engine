# Half-season chip plan

Locked with Gemini on 2026-10-04 ([half plan](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). Nothing in this note is coded yet. `score_xp` stays the published score. A double stays one fixture. The historical climb keeps an empty chip map.

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

Phase 1 is a pure function, `plan_half`, in `src/live/half_plan.py`, with tests in `tests/test_half_plan.py`. The caller passes the week tables. The function returns the schedule and the chip for this week. It does not replace `recommend_chip`. It does not call the squad picker, the climb, or the network. The first code commit touches those two files only.

Phase 2, after those tests, changes the transfer search on one point. If the planned Bench Boost week sits inside the next three weeks, that week's bench is scored at `score_xp` and every other week's bench stays at zero. The weight is not a constant.

Phase 3 is the part that keeps a strong bench for a Bench Boost outside those three weeks, then drops the term once the chip is used, so later transfers sell the bench down. Its formula is locked only after phase 1. A historical climb that fills in the chip map is not part of phase 3. That climb would move published season totals and needs its own lock.

## What phase 1 does not see

Transfers between this deadline and the Bench Boost week are invisible until the next solve. The wildcard fifteen is whatever the caller priced. The Gameweek 6 example is in the sum: the wildcard's value includes the Bench Boost it makes available before Gameweek 19, and the weeks after that chip do not pay for the bench. The transfers that actually assemble and then sell that bench are phases 2 and 3.
