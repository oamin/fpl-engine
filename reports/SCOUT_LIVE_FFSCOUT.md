# Live scout — FFScout minutes packet (v1)

**Started:** 2026-11-01  
**Module:** `src/live/scout_minutes.py`  
**Lock:** shares freeze under scout m — [`SCOUT_M_SHARE_FREEZE_LOCK.md`](SCOUT_M_SHARE_FREEZE_LOCK.md)

## Contract

| Layer | Rule |
| --- | --- |
| **Default (everyone)** | Expanding as-of mean `m` (GW `< t`), flat across the horizon |
| **Scout overlay (sparse)** | FFScout **Out / Banned / Doubts** → step-0 anchor `m0` |
| **Horizon return** | Decay to expanding: `m(h) = expand + (m0 − expand) · ρ^h` (default **ρ=0.85**) |
| **Hard pins (later)** | Structured `out_for_n_gws` can force early steps via `force_steps` |
| **Shares** | Freeze `sG`/`sA` from expanding baseline when scout minutes are applied |
| **Not scout (yet)** | Blanket firm-XI minutes overrides; prose-parsed return length |

Predicted XI still drives the **selectable pool** (WHO), not the minutes overlay.

## Build

```bash
# from saved FFScout snapshot (offline)
PYTHONPATH=. python3 -m src.live.scout_minutes --gw 9 \
  --from-json data/scratch/ffscout_gw09/ffscout_team_news.json --smoke

# live fetch
PYTHONPATH=. python3 -m src.live.scout_minutes --gw N --fetch
```

Writes `data/predictions/2026-27/gwNN/scout_minutes.json` (+ optional `scout_xmi_week0.csv`).

## Live pick hook

```bash
python -m src.live.numagent_live_pick --gw N --scout-packet \
  data/predictions/2026-27/gwNN/scout_minutes.json …
```

Week-t uses step 0; t+1 uses step 1 (Out revert to expanding). Share freeze on.

## GW9 smoke (saved FFScout page)

- Baseline players: ~669  
- Overlays: Out / Banned / Doubts (~89 after name match)  
- Saka example (expand≈69′, ρ=0.85): GW9 **0′** → GW10 **≈10′** → GW11 **≈19′** → … → expanding  

## Context labels (combined)

`src/live/scout_context.py` enriches overlays with human labels:

| Label | Signals |
| --- | --- |
| `injured, expected back in X weeks` | FPL “Expected back …” date and/or blurb (`next N Gameweeks`, `until GW N`, `a few weeks`, internationals) → `force_steps` pins |
| `injured, unknown return` | Out with no horizon / long-term absentee / past-due FPL date |
| `suspended, expected back in X weeks` | Banned + ban-length blurb / FPL suspended-until |
| `doubt (P%)` | FFScout Doubts |
| `suspected single-week rotation` | Regular (expand ≥45′) left out of predicted XI, not Out/Banned/Doubt |
| `sustained rotation` | Same + last 3 GWs all ≤30′ |

Example inspect: [`scout_context_example_gw09.md`](scout_context_example_gw09.md) · JSON highlights [`scout_context_example_gw09.json`](scout_context_example_gw09.json).

## Next

- Tune XI-omit false positives (Raya/Alisson when cup XI etc.)  
- Opta-far horizon boards consume per-step maps  
- Coach brief consumes overlay list only
