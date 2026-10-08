# Betfair live lines pull (2026-10-08)

Target: `data/predictions/2026-27/gw06/betfair_20261008/` via pure Exchange on `refresh_lines` + prop fetch. Pulled on the Mac (allowed geo). `.env` mode 600, gitignored. `data/live/` freeze untouched.

## Ladder fix

First pass authenticated but wrote the neutral 2.5 / 3.70 / 3.03 prior for every match: `listMarketBook` with `EX_BEST_OFFERS` returns the ladder under `runner.ex`, and `best_prices` had been reading the runner root. The reader now prefers `ex`, with a root fallback. Unit tests cover both shapes.

## Result (second pull)

| Item | Status |
| --- | --- |
| Gameweek 6 MATCH_ODDS | Exchange prices: 5 tier 1, 5 tier 2; every match has OU 2.5 |
| Arsenal–Leeds | 1.42 / 5.36 / 9.31; OU 2.5 at 1.93 / 2.07 |
| Gameweek 7 MATCH_ODDS | Neutral prior (9 thin; Nottingham Forest–Arsenal no two-sided mid); 9/10 still have OU 2.5 |
| TO_SCORE (anytime) | Empty — one market (Arsenal–Leeds), 0 of 41 runners cleared £250 matched + spread |
| Outrights | 19 clubs in `outrights_ranks.json` |
| BTTS | Diagnostic only; probability on 19 of 20 matches |

Imminent goals stay `share_xG × λ` until anytime liquidity clears the gate. Discovery for gameweek 6 picks up this folder, so GW6 match odds and outrights enter `score_xp` / `forecast_xp`.

## Session token

The session token was pasted into an earlier chat. If the transcript is shared, log in again on Betfair and replace `BETFAIR_SESSION_TOKEN` in the local `.env`. Do not commit `.env`.
