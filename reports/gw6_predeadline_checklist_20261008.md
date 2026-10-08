# GW6 pre-deadline checklist

Gemini ([checklist](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): sensitivity audit is done; nothing else to invent tonight. Remaining work is input refresh + T−1h commit on the Mac.

## Already closed

- Stack audit (tags / Betfair / outrights)
- Ladder fix + Mac Betfair sync
- Horizon / outrights design; no fresh MILP before T−1h
- Odds API off the live path

## Friday (after pressers)

1. Refresh `xmi_gw6.csv` via the minutes completion on a fresh bootstrap.
2. Optional: dry-run `python3 -m src.live.wildcard_plan --capture <latest official csv>` to check the knapsack still fits after price changes.

## Saturday T−1h (Mac only)

1. Capture `slot_t1` / official `ep_next` (capture schedule job).
2. `python3 -m src.live.betfair_lines --out data/predictions/2026-27/gw06/betfair_t1`
3. `python3 -m src.live.wildcard_plan --capture <that official file>` → log chip decision.
4. Write the paired shadow + deadline note (`deadline` / capture path as wired).

## Do not

- Re-pull Betfair before Friday evening
- Odds API
- Cloud MILP / multi-period transfer MILP
- Retune gates or formula
- Forum/Twitter manual score edits
