# Three hypotheses

An outside note asked whether the early season, the shape of a player's points, or the transfer search explains the gap to strong managers. The audit below uses the code and the tests already run. Nothing in the engine is changed. Gemini locked the two tests that are still worth running ([hypothesis audit](bc-7121b96b-db3a-552d-ade8-335c01d37eda)).

## Transfer search

The search is not limited to one sale. `choose_transfers` tries one, two, and three transfers. A one-transfer move is same-position only, the top 35 by this week's score, then the best six by the three-week value. A second transfer starts from one of those six. A third is tried only after a two-transfer move is already the best. The simultaneous cross-position list exists and is off on the published path. That list keeps a pair only when the two legs together gain score, and it only looks at the weakest four owned players and the top eight buys at each position.

Stage 44 already priced the pairs that list contains and the beam does not try. On 2025/26, Gameweeks 5–38, they were worth 0.06 of three-week value a week. No week reached 1. The largest week was 0.77. That is value, not points scored. See `reports/stage_44_search_counts.md`. The price-reach count on this season's first five weeks is a different fact: six human targets became affordable with a second sale, and those six summed to −7 points. That was not the search's objective, and it did not open a replay.

What is still unknown is whether 2025/26 was a quiet season. The same shadow has not been run on 2022/23, 2023/24, or 2024/25. An exhaustive listing of every legal pair is not the next test. The shadow is the cash-release path the note describes.

The test reruns that shadow. No search setting changes. A rewrite is rejected if each of those three seasons averages under 0.25 value points a week and has fewer than three weeks at or above 1. A season above that is inconclusive, and the search still stays as it is. The 2026 cohort is not the sample. The data are the cached season sheets. One season of this shadow has already been run, so three more are the same job. This should be a pull request only after the three seasons are counted. No implementation change now.

## Cold start

At a deadline the score uses last week's information, shifted by one week. Minutes are a rolling three. Shots are an expanding mean. The current match is priced from that week's opening 1X2 and the 2.5 line, so the book for the fixture is in the score. A new signing cannot be bought until three appearances and 45 expected minutes. An owned player with one or two appearances keeps a past-only score capped at 6.

The historical season builder does not carry a player across seasons. Gameweek 1 therefore has no earlier row, and a missing minutes or shot figure is filled with the position mean of the whole season, including later weeks. That is leakage. The live path does not do it. Last season is the only fill, and a matched player keeps last season's shifted prior. A debutant gets the position mean of those matched first rows.

The score does not contain a manager, a tactical note, set pieces, preseason, a predicted eleven, or an injury flag. Historical status was parked because the sheets do not record it before the deadline. There is no 2021/22 sheet in the cache, so a clean fill can be built for 2023/24, 2024/25, and 2025/26 only. The published fast eleven starts at Gameweek 5, because earlier weeks do not have three appearances inside one season. There is no archive of strong human squads for those seasons. Crowd Gameweek 1 fifteens exist and are a different comparison.

The 2026 Gameweeks 1–5 were a score miss and a budget limit. They are not the cold-start test.

The test is an information-clean fast eleven, previous season as the only fill, on those three seasons. Cold start is kept only if, in at least two of the three, the clean eleven in Gameweeks 1–8 scores at least 2 points a week less than the same eleven in Gameweeks 9–38, and the clean eleven in Gameweeks 5–8 scores at least 1 point a week less than the published eleven in those same weeks. Otherwise the hypothesis is rejected. No model change inside the test. The published comparison can only share Gameweeks 5–8, because the published eleven does not exist before that. This runs after the search shadow. It is a new measurement, not a change to the live score. No implementation change now.

## Distribution

Two players on the same expected points can have different weeks. The decision uses the sum only. The formula does store the chance of appearing, the chance of 60 minutes, expected goals, and the clean-sheet probability, and then adds them. It does not store a quantile or a correlation between players.

Treating spread as a penalty has already been run. Dividing the score by the spread lost 253 points. Subtracting a quarter of the spread lost the fast eleven by 100. Using the spread only inside transfers changed sign across seasons. This morning's count of earlier blanks and hauls, on players within half a point, was short of 80 decisive pairs in every season and missed a 5 point margin over the leader in three seasons. Both shapes are parked.

Under expected points, the captain is twice one player's points. The player with the higher expected score is the captain that maximises the expected total. Drawing a random week from a distribution around that mean adds noise and, on average, picks a worse captain. The same arithmetic covers Triple Captain. Correlation between teammates changes the spread of a squad's week. It does not change the expected total, which is still the sum of the means. A distribution would matter if the objective stopped being expected points. That is a different objective, and the spread penalties that were tried as that kind of objective lost.

No further distribution test is queued. No code change.

## Upper tail, reviewed and not queued

A later note accepted that a generic spread penalty should stay parked, and asked a narrower question. When two players are within half a point, does a higher historical ceiling (the 90th percentile, or the rate of weeks at 10, 12, or 15 points) predict more points than the slightly higher score? If it does, the mean is missing a feature. If it does not, the hypothesis is dead.

That question is the haul arm already counted this morning. The feature was the share of earlier weeks at 8 points or more, on the same pairs, the same half-point gap, four seasons, shift-1. Decisive pairs were 57, 56, 60, and 60. The margins against the leader were +7.5, −4.3, +4.2, and −1.6 percentage points. It missed +5 in three seasons and missed 80 pairs in every season. See `reports/close_pair_shape.md`.

A rarer cut has fewer events, not more. A week of 10 or more is a subset of a week of 8 or more, so the decisive pairs would shrink below the floor they already missed. Running 10, 12, and 15 together would be a grid after a failed test. Gemini declined that ([upper tail](bc-7121b96b-db3a-552d-ade8-335c01d37eda)).

The stacking point is already inside the mean. A goal and the bonus it tends to bring are added as 0.18 times expected goal points. On 7,569 buy-pool rows that bonus proxy has a mean error of −0.082, inside the leave-alone band. Dependence changes how wide a single week is. The expected total is still the sum of the means, and the captain who maximises that total is still the higher expected score. A ceiling would change the captain only if it predicted the mean better than `score_xp`. The haul arm was that check, and it did not.

## Order

The search shadow on the three earlier seasons is first. The clean-fill fast eleven is second. The upper-tail reading is not added. Nothing is run in this note.
