# Item 2 — Minutes / availability overlay

Graded FPL availability on the existing selectable pool (not a second hard
filter). Scales `xmi`, `share_xG`, `share_xA`, and `exp_defcon_hit` by
`chance_of_playing_next_round` (fallback: this-round / status defaults).

Injured / suspended → 0; doubtful without a % → 0.5; chance 75% → ×0.75.

## Pool impact (GW6 bootstrap)

| | |
| --- | ---: |
| Selectable | 180 |
| Scaled (`factor < 1`) | **19** |
| Zeroed | **4** (3 injured, 1 suspended) |
| Mean factor | 0.957 |
| Status mix | a=161, d=15, i=3, s=1 |

Examples zeroed: Kluivert, Pau, Dedić, Fatawu. Doubtful at 0.75×: Rice, White, Havertz, Palmer, Brobbey, Dunk, …

## Cascade diagnostic (same seed as FT path)

`xi_plus_captain_ft`, N=6, snapshot_every=2, seed=7.

| Player | freq no overlay | freq overlay | Δ |
| --- | ---: | ---: | ---: |
| Brobbey (d 75%) | 0.17 | 0.06 | **−0.11** |
| Khalaili (d) | 0.28 | 0.06 | **−0.22** |
| Hume | 0.50 | 0.19 | **−0.31** |
| Hall | 0.72 | 0.67 | −0.06 |
| Mbeumo | 0.64 | 0.69 | +0.06 |
| Saka | 0.67 | 0.69 | +0.03 |
| Haaland | 0.64 | 0.61 | −0.03 |

Mean transfers 2.2 / hits 0.93 (slightly more churn as doubtfuls are sold).

## Read

1. Overlay **does what it should**: doubtful/injured pool members lose frequency; fit replacements rise.
2. Effect is **local** (tens of players), not a reshuffle of the elite board — Haaland/Saka barely move.
3. Pen / set-piece **role** bump still deferred (no clean prior on disk).
4. Default **on** for multistage CLI (`--minutes-overlay` / `--no-minutes-overlay`).

## Reproduce

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp \
  --n 6 --snapshot-every 2 --seed 7 --objective xi_plus_captain_ft \
  --no-minutes-overlay --out data/predictions/2026-27/multistage_ft_no_overlay
PYTHONPATH=. python3 -m src.models.multistage_mc_milp \
  --n 6 --snapshot-every 2 --seed 7 --objective xi_plus_captain_ft \
  --minutes-overlay --out data/predictions/2026-27/multistage_ft_overlay
```
