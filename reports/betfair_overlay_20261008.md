# Betfair odds → score_xp and forecast_xp

Gemini (bc-e75c8209): KEEP MATCH_ODDS + OVER_UNDER_25 as the live pot; CHANGE TO_SCORE into imminent-week score_xp under the team-λ cap; CHANGE season outrights into shrunk strength priors for unpriced forecast_xp weeks; DROP First Goalscorer, BTTS (diagnostic only), Correct Score, and Half-Time from player scoring.

## Rename

Look-ahead lives in `src/models/forecast_xp.py`. Primary names: `compute_player_forecast`, `make_forecast_steps`, `attach_forecast_xp`. `src/models/open_horizon.py` re-exports the old names.

## Storage (valuable markets)

| Artifact | Role |
| --- | --- |
| `gw_lines.csv` | MATCH_ODDS + OU 2.5 → pot for score_xp / forecast_xp |
| `betfair_to_score.json` | Anytime goalscorer → imminent score_xp goals |
| `outrights_ranks.json` | Winner / top 6 / relegation → unpriced forecast_xp pots |
| `diagnostics_btts_cs.json` | BTTS diagnostic only |
| `data/scratch/betfair/*` | Raw books (gitignored) |

## Use

`price_half(..., artifacts_dir=...)` loads the derived JSON beside the lines file. Imminent week: TO_SCORE rates replace `share_xG × λ` when matched. Unpriced later weeks: outright strength pots at κ=0.5 shrinkage; else copy the last priced step.

## Pull

```bash
python3 -m src.live.betfair_lines --out data/predictions/2026-27/gw06/betfair_20261008
```

Must run from an allowed geo (US cloud gets HTTP 403). Historical `compute_xp` / published `score_xp` on closed seasons are unchanged.
