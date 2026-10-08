# Betfair live lines pull (2026-10-08)

Target: `data/predictions/2026-27/gw06/betfair_20261008/` via pure Exchange `MATCH_ODDS` + `OVER_UNDER_25` on `refresh_lines`.

## Result

**Blocked: Betfair geo (HTTP 403).** This agent run was placed on a US cloud VM (Ohio / AS16509), not the PI MacBook private worker. Session token and app key were present in gitignored `.env`; `BETFAIR_USERNAME` / `BETFAIR_PASSWORD` were empty, so no SSO re-login was attempted. Keepalive and `listMarketCatalogue` both returned the Betfair geo HTML 403 page (not an auth JSON error).

FPL bootstrap + fixtures were refreshed into the predictions dir only (`data/live/` freeze untouched). Lines CSV stayed header-only (`rows=0`, `reason=betfair_geo_blocked`).

## Re-dispatch

MacBook worker was connected and idle at pull time:

- `workerId`: `fa332b94-0afc-56cb-a773-b70179980541`
- `machineDisplayName`: `MACBOOK-YRQYDLWWPR`
- `eligibleForSubagent`: true

Re-run the same pull on that worker with `usePrivateWorker=true`. Secrets stay in `.env` (never commit).

## Code on this branch

Working tree wires `src/live/lines.py` to Betfair only (Odds API / ESPN removed from the live path). Unit tests: `tests/test_betfair.py` + `tests/test_live_lines.py` (19 passed).
