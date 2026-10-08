# Weekly freeze ledger and multi-week bar

Gemini ([freeze bar](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): lock the schema and bar **before** GW6 points; write the first ledger row only at the T−1h decision. No stub rows in the JSONL.

## Module

`src/live/freeze_week.py` → append-only `data/predictions/2026-27/week_freeze.jsonl`.

- `write_deadline_freeze` — one row per gameweek, realised nulls.
- `attach_realised_outcomes` — Monday fill of the `realised` block only; forecast/decision/provenance immutable (`ProvenanceViolationError` otherwise).
- Pool forecasts are stored by **sha256**, not the full player table.

## Bar (aligned with `decision_spec`)

| Item | Value |
| --- | --- |
| Window | GW6–GW25 (N = 20 = `MIN_LIVE_WEEKS`) |
| First formal read | GW26 (`LIVE_REVIEW_GW`) |
| Continue if undetermined | GW38 (`LIVE_CONTINUE_TO_GW`) |
| Squad metric | Σ (actual decision XI − actual 1FT XI); hits already in those totals |
| Squad pass | cumulative net > 0 with 95% block-bootstrap interval excluding 0 |
| Forecast metric | encompassing: points ~ ep_next + score_xp on the frozen pool |
| Forecast pass | β(score_xp) > 0 at p < 0.05; else undetermined |

Interim week totals are monitoring only. They do not retune the model.

## T−1h write (Mac)

After `slot_t1`, Betfair pull, minutes, and `wildcard_plan` decision:

```bash
# Caller assembles the row (hashes + XI lists) then:
python3 -c "from src.live.freeze_week import write_deadline_freeze, build_freeze_row; ..."
```

Or wire a thin CLI later. Do not append a GW6 row before that window.

## Do not

- Touch `data/live/HOLDOUT_FREEZE.json`
- Put Odds API back on the live path
- Commit mock ledger rows before T−1h
