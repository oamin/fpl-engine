# Betfair exchange overlay (diagnostic + live ingest)

Session token and app key are in gitignored `.env` (never printed).

Gemini (bc-e75c8209): **CHANGE** the live pot ingest to pure Betfair `MATCH_ODDS` and `OVER_UNDER_25` using simplex-normalised mid probabilities with tiered liquidity shrinkage; **KEEP** anytime-goalscorer as a live-only overlay with the team-λ cap; **KEEP** clean sheet and BTTS diagnostic only; **DROP** Odds API fallbacks, player assists, and outright ratings from the active decision plan.

## Live path

`src/live/lines.refresh_lines` is Betfair-only. Fair decimals are `1/p` after simplex, so `side_pot`'s Shin step is the identity on fair books. Tiered liquidity:

- Tier 1: matched ≥ £25k and relative spread ≤ 0.10 → pure mid
- Tier 2: £5k–£25k → shrink toward (0.40, 0.27, 0.33)
- Tier 3: thinner / wider → neutral pot, tagged

Command (run from an allowed geo — not a US cloud VM):

```bash
python3 -m src.live.betfair_lines --out data/predictions/2026-27/gw06/betfair_20261008
```

Frozen `data/live/gw_lines.csv` is not overwritten.

## Pull status (this environment)

Blocked: Betfair geo HTTP 403 from the US cloud egress. MacBook worker `fa332b94-0afc-56cb-a773-b70179980541` was connected and eligible; Task placement could not force that worker from this run. Re-run `python3 -m src.live.betfair_lines` on the MacBook (or any allowed IP) with the same `.env` keys.
