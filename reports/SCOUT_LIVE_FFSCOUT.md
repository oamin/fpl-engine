# Live scout — FFScout minutes packet (v1)

**Started:** 2026-11-01  
**Module:** `src/live/scout_minutes.py`  
**Lock:** shares freeze under scout m — [`SCOUT_M_SHARE_FREEZE_LOCK.md`](SCOUT_M_SHARE_FREEZE_LOCK.md)

## Contract

| Layer | Rule |
| --- | --- |
| **Default (everyone)** | Expanding as-of mean `m` (GW `< t`), flat across the horizon |
| **Scout overlay (sparse)** | FFScout **Out / Banned / Doubts** only → step-0 minutes; later steps revert to expanding until structured `out_for_n_gws` exists |
| **Shares** | Freeze `sG`/`sA` from expanding baseline when scout minutes are applied |
| **Not scout (yet)** | Blanket firm-XI minutes overrides; multi-week injury length from prose |

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
- Overlays: Out / Banned / Doubts (~87 after name match)  
- Step-1 for Out = expanding baseline (not sticky-zero)

## Next

- Structured `out_for_n_gws` / permanent rotation tags (news + FFScout blurb)  
- Opta-far horizon boards consume per-step maps  
- Coach brief consumes overlay list only
