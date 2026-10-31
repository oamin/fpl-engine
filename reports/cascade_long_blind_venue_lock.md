# cascade_long club λ — blind Opta locked as default

**Locked:** 2026-10-29 (PI)  
**Protocol:** `cascade_long_match_outcome` · `cascade_long_blind_venue_lock`  
**Module:** `src/models/opta_gf_ga_horizon.py`

## Defaults (locked)

| Constant | Value | Note |
| --- | --- | --- |
| `DEFAULT_VENUE_MODE` | `blind` | Season-total place GF/GA → λ; no place×venue remap |
| `AWAY_SHRINK` | `1.0` | Identity on λ_A |
| `LAM_SHRINK` | `1.0` | Identity on both λ |
| `HOME_ADV` | `1.0` | Deprecated alias; no symmetric spice |

## Why

PI confirmed after the Opta−FD residual dig chain: **blind Opta is best left as-is**.

Screened remaps either failed OOS (≤2/4 seasons) or hurt MAE vs actual:

| Arm | Outcome |
| --- | --- |
| place×venue / `venue_away` | holdout PASS; OOS **1/3** |
| expanding club H/A | holdout away PASS; OOS **2/4** |
| FD venue mean-ratio | holdout away_only PASS; OOS **2/4** |
| margin-shrink α<1 | inert under blind (E[λ_H]≈E[λ_A]); OOS **0/4** |
| fixed `away_shrink` / `lam_shrink` ≠1 | OOS **0/3** or home |bias| worsen |
| lag-1 Opta−FD | tracks book, not ball |
| lag-1 Opta−actual | MAE worse |
| remaining-season balance | MAE **1.047** vs blind **0.949** |

Sticky Opta−FD venue×quality residual remains a book-vs-place feature, not a stable OOS club-λ fix.

## Re-open rule

Dated protocol update + new pre-registered screen with OOS ≥3/4 and PI re-approval before flipping `venue_mode` or identity shrinks.
