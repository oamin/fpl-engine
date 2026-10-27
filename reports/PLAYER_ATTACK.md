# Player attack from `M` — pots × shares

Part of the `score_xp_v2` decomposition. Module: `src/markets/player_attack.py`.

Upstream: [`MATCH_OUTCOME.md`](MATCH_OUTCOME.md) builds the Dixon–Coles grid `M` and side moments (`xG`, `xCS`, `xGC`). This module turns **one side’s `xG_M`** into player goal / assist expectations via compositional shares.

## Boundary

| In this module | Not here |
| --- | --- |
| `G_team = α_G · xG_M`, `A_team = α_A · xG_M` | Fitting `M` |
| `E[G_i] = sG_i · G_team`, `E[A_i] = sA_i · A_team` | Building `rG`/`rA` or shares |
| Optional anytime **props → π** (goals only) | Absolute `λ` pin / lift-only |
| Optional FPL `xp_goals` / `xp_assists` | Appearance, CS, GC, DefCon, bonus |

Shares stay in `numagent_xp` (rates × minutes → weights → share-SE shrink). The assembler `score_team_side` calls this module for the attack block only.

### Anytime props (row 1 — locked formula; live default still **off**)

Props **shape** goal shares; `M` **levels** them:

```
w_i = λ(p_i) if covered else sG_i · G_team
π_i = w_i / Σ w
E[G_i] = π_i · G_team          # mean path
E[G_i | k] = π_i · k           # short-horizon roll (player_attack_on_roll)
```

Assists: **no** prop overlay (locked). Lift-only and hard absolute replace are parked.

## API

```python
from src.markets.player_attack import player_attack_from_m

pots, exp, prop_diag = player_attack_from_m(
    sG, sA,
    xg=xg_m,           # E[goals for] from M
    alpha_g=0.96, alpha_a=0.90,
    positions=["MID"], # optional → xp_goals / xp_assists
    prop_p={0: 0.45},  # optional row-index → P(anytime)
)
# pots.G_team, pots.A_team
# exp.E_G, exp.E_A  ← player xG / xA in the score_xp_v2 sense
```

## Worked example — Saka, GW20 2025/26

**Fixture:** Bournemouth vs Arsenal (Arsenal away), 2026-01-03.  
**`M` moments (FD AH+OU):** `xG_M = 2.079`, `xCS = 0.380`, `xGC = 0.270`.  
**α (pooled):** `α_G = 0.9601`, `α_A = 0.9050`.

```
G_team = 0.9601 × 2.079 = 1.996
A_team = 0.9050 × 2.079 = 1.881
```

**Saka rates (same-season, GW < 20, appearances only):** 17 apps, 1320′, 4 G, 6 A, `M0 = 270`

```
rG = 4 / (1320+270) = 0.002516
rA = 6 / (1320+270) = 0.003774
m  = mean(last 3 GW mins) = 87.3
wG = rG = 0.002516          →   sG (shrunk) among m>0 club  (π∝r; locked 2026-10-27)
wA = rA · m/90 = 0.00366    →   sA (shrunk) = 0.159
```

**Player attack (this module):**

```
E[G] = 0.131 × 1.996 = 0.261   →  xp_goals = 0.261 × 5 = 1.31
E[A] = 0.159 × 1.881 = 0.298   →  xp_assists = 0.298 × 3 = 0.89
```

Among Arsenal’s playing set that week he ranks **#3 in `sG`** (behind Gyökeres, Trossard) and **#1 in `sA`**. Realised that GW: 23′, 0 G, 1 A, 4 pts (rotation — oracle had ~87′).

## Decomposition status

| Component | Status | Module |
| --- | --- | --- |
| `match_outcome_probs` / `roll` | defined | `match_outcome.py` |
| **Player attack from `M`** | **defined** | **`player_attack.py`** |
| Rates / shares / α | locked | `numagent_xp` |
| Defence from `M` (CS / GC) | still in assembler | `score_team_side` |
| Assembler | locked; attack block delegates here | `numagent_xp` |
