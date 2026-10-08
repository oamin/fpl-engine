# Betfair exchange overlay (diagnostic)

Pulled at `20261008T122156Z`. App key loaded from the environment; not printed.

Gemini (bc-e75c8209): DROP outright-derived ratings from the active decision horizon and keep unpriced weeks unknown; CHANGE Betfair match and goalscorer overlays to require two-sided liquidity and simplex normalisation before entering shadow diagnostics.

Auth: Betfair needs BETFAIR_USERNAME and BETFAIR_PASSWORD (or BETFAIR_SESSION_TOKEN) in the environment

No books were pulled. Set `BETFAIR_USERNAME` and `BETFAIR_PASSWORD` (or a fresh `BETFAIR_SESSION_TOKEN`) in `.env` alongside the app key, then re-run `python3 -m src.live.betfair_pull`.
