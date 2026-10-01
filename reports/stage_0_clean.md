# Stage 0 — Clean slate

## Removed
- Old analyse / λ / DefCon / match_intensity / parse_vaastav scripts
- `odds_data.py`, `build_match_odds_points.py` (replaced by `src.ingest`)
- Derived CSVs, plots, regimes, obsolete matrix Cursor rule

## Kept
- `.env.example`, `.gitignore`, `.venv`, `uv.lock`
- `.cursor/rules/odds-api-quota.mdc`
- `data/cache/odds_snapshot.json` (no Odds API re-fetch)

## New layout
- `src/ingest/` — FPL + football-data odds
- `src/features/` — shares, baseline
- `src/models/` — team market eval, player μ
- `src/markets/shin.py` — Shin de-vig
- `reports/stage_*.md` — mini-reports
- `data/processed/` — pipeline outputs

## Next
`uv run python -m src.run_frontend`
