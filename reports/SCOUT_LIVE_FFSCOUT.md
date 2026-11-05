# Live scout — FFScout minutes packet (v1)

**Started:** 2026-11-01  
**Module:** `src/live/scout_minutes.py`  
**Lock:** shares freeze under scout m — [`SCOUT_M_SHARE_FREEZE_LOCK.md`](SCOUT_M_SHARE_FREEZE_LOCK.md)

## Contract

| Layer | Rule |
| --- | --- |
| **Default (everyone)** | Expanding as-of mean `m` (GW `< t`), flat across the horizon |
| **Scout overlay (sparse)** | FFScout **Out / Banned / Doubts** (+ XI-omit rotation) → known-horizon pins |
| **Horizon return** | **Pin then snap** — no ρ-decay. `force_steps` / `return_gws` / step-0 `m0`; after that → expanding `m` |
| **Hard pins** | Structured `out_for_n_gws` / FPL+blurb return horizon via `force_steps` |
| **Shares** | Freeze `sG`/`sA` from expanding baseline when scout minutes are applied |
| **Role** | Closed tags on full pool + overlays: `firm_starter` / `tactical_starter` / `rotational` / `impact_sub` / `cup_starter` / `fringe` (`src/live/scout_role.py`) |

Predicted XI still drives the **selectable pool** (WHO); XI-omit also soft-pins week-0 minutes for regulars left out. Role is the playing pattern when available (separate from injury context).

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

Week-t uses step 0; t+1 uses step 1 (snap to expanding unless still pinned). Share freeze on.

## GW9 smoke (saved FFScout page)

- Baseline players: ~669  
- Overlays: Out / Banned / Doubts (+ XI-omit) after name match  
- Cherki (single-week XI omit, expand≈55′): GW9 **15′** → GW10+ **expand** (snap, no climb)  
- Known return (e.g. 2 GW out): pin zeros for steps 0–1 → snap to expand at step 2  


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

**Rotation tune trial (GW9):** XI→id resolve + match-quality gate + GK guard + rest/cup veto. Rotation labels 36→25; precision vs actual minutes 61%→80%. Report: [`scout_context_tune_trial_gw09.md`](scout_context_tune_trial_gw09.md).

**Partner blurb (committed):** within-season same-position anti-corr teammate text on overlays (`src/live/minutes_partner_context.py`). Context only — **no minutes rating**. Dig: [`minutes_anticorr_role_dig_2026_27.md`](minutes_anticorr_role_dig_2026_27.md).

## Next

- Further cut remaining FPs (Guéhi/Botman-style nailed CBs on alternate XI)  
- Opta-far horizon boards consume per-step maps  
- Coach brief consumes overlay list only
