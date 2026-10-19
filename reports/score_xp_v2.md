# `score_xp_v2` — comprehensive report (current form)

**Status:** locked as Numagent 2.0’s **primary** player-GW scorer (2026-10-13; GKP saves multinomial locked 2026-10-14).  
**Does not replace** published `score_xp` on free-transfer climbs or protocol comparisons.  
**Assembler:** `src/markets/numagent_xp.py` (`SCORE_NAME = "score_xp_v2"`).  
**Scoring constants:** `src/rules/fpl_2026.py` (never duplicated elsewhere).  
**Shorter method lock:** `reports/numagent_2_0_method.md`.  
**Gated screen:** `reports/score_xp_v2_gate.md`.

This document describes **exactly what is implemented** today: every term, every gate, every position rule, and how the score behaves by position on closed data.

---

## 1. Role

| Item | Value |
| --- | --- |
| Score column | `score_xp_v2` (also written as `xp`) |
| Uncertainty column | `score_xp_v2_se` / `xp_se` |
| Development metric | Player-GW Gaussian log-score, σ = 3, on the **point mean** |
| `xp_se` role | Stability / reliability of the assembled mean — **not** predictive σ for points |
| Historical screens | Minutes **oracle** (actual `m`, `σ_m = 0`) |
| Live path | Minutes from tags / `xmi` (+ `xmi_sigma`); same assembler. **Scoreline (PI):** include **BTTS when available**; accept **non-2.5 OU** (3.0, 3.5, …) and arbitrary AH lines (±3, ±3.5, …) — see §2.1 |
| Holdout | 2026/27 frozen — never fit on, never in closed comparisons |

---

## 2. End-to-end pipeline

```
AH + OU 2.5  →  Dixon–Coles matrix M  →  (xG, xCS, xGC)
                                              ↓
                         α_G, α_A  →  G_team, A_team
                                              ↓
         player rates rG/rA + minutes m,σ_m  →  shares sG/sA
                                              ↓
                    E[G], E[A] + defence/prior terms  →  score_xp_v2
```

### 2.1 Scoreline matrix `M`

- Fit: Dixon–Coles to fair market targets (`src/markets/scoreline.py`). Historical closed screens use **AH + OU 2.5** with BTTS weight **0**.
- Historical odds: football-data **opening** AH + OU 2.5 only (no BTTS on free FD history).
- **Live weeks (PI requirement — implement / wire when live scoreline is next touched):**
  1. **BTTS:** include in the fit **whenever a fair BTTS price is available** (weight > 0). Fall back to weight 0 only if the market is missing or illiquid — do not leave BTTS permanently off on live.
  2. **OU line flexibility:** do not hard-require OU **2.5**. When the book’s main or usable total is **3.0, 3.5, 4.0, …** (or another listed total), fit `P(goals ≥ line)` against that line. Prefer the most liquid / main total; if several are present, document the selection rule in the live pull.
  3. **AH line flexibility:** already parameterized by `ah_line` (including large lines such as **±3, ±3.5**); live must pass through whatever AH point the book quotes, not assume −0.5 / 0 only.

Assembler moments (xG / xCS / xGC) are unchanged once `M` is fit; only the **targets and weights** into `fit_scoreline` change on live.

### 2.2 Team moments (`scoreline.team_moments`)

For home (swap indices for away):

| Moment | Definition |
| --- | --- |
| **xG** | `E[goals for]` from `M` |
| **xCS** | `P(GA = 0)` = sum of `M` over opponent score 0 |
| **xGC** | `E[⌊GA / 2⌋]` — GC **events**, not `⌊E[GA]/2⌋` |

### 2.3 FPL award scalings α

Store: `data/processed/fpl_goal_assist_rates.json` · code: `src/markets/fpl_score_rates.py`.

| Symbol | Meaning | Prior 2025/26 (clean) |
| --- | --- | ---: |
| α_G | FPL player goals / scoreline match goal | 0.9615 ± 0.006 |
| α_A | FPL assists / scoreline match goal | 0.9027 ± 0.010 |

```
G_team = α_G × xG
A_team = α_A × xG
```

**α cleaning only (not the player allocator):** when estimating α from history, drop fixture-sides where Σ FPL-awarded player goals > scoreline match goals for that side (data bugs / own-goal quirks). That filter does **not** apply to `E[G_i] = sG_i × G_team`: shares sum to 1, so Σ E[G_i] = G_team by construction and no player forecast can exceed the team pot in aggregate.

**Live seasons:** after each completed GW, add clean counts to the prior and refresh pooled α (do not freeze the prior point estimates).

### 2.4 Player attack rates (as-of)

Built by `player_rates_from_history` on history with **GW < t** only:

- Appearance rows only: `minutes > 0` (injury blanks excluded).
- `rG = Σ goals / Σ minutes`,  `drG = √(Σ goals) / Σ minutes` (0 if no goals).
- `rA`, `drA` analogously for assists.
- **No** soft Bayesian shrink toward a position prior.

### 2.5 Minutes

| Context | `m` | `σ_m` |
| --- | --- | --- |
| Closed / gated screens | Actual minutes | 0 |
| Live | Tag / `xmi` regime mean | From tag rules + 3-GW history (`xmi_sigma`) |

If `m ≤ 0`, **every** term is zero (player out).

### 2.6 Weights and shares

```
w = r × m / 90
Δw = √( (m/90 · Δr)² + (r/90 · σ_m)² )
s = w / Σw          (within the scored team-side table)
```

Share SE treats the sum as dependent (method §6 in `numagent_2_0_method.md`). Separate share vectors for goals (`sG`) and assists (`sA`).

### 2.7 Attack expectations

```
E[G_i] = sG_i × G_team
E[A_i] = sA_i × A_team
xp_goals   = E[G_i] × GOAL_POINTS[pos]
xp_assists = E[A_i] × ASSIST_POINTS          # ASSIST_POINTS = 3 for all positions
```

Relative SE on `E` combines share SE and α SE in quadrature when provided.

---

## 3. Total identity

```
score_xp_v2 =
    xp_appear
  + xp_goals
  + xp_assists
  + xp_cs
  + xp_gc
  + xp_defcon
  + xp_bonus
  + xp_cards
  + xp_saves
```

```
xp_se = √( Σ_components (Δ component)² )
```

Components in the quadrature: goals, assists, CS, GC, DefCon, bonus, cards, saves.  
Appearance has no SE term. Own goals and penalty miss/save stay at **0**.

---

## 4. Component methods (as implemented)

All constants below are from `src/rules/fpl_2026.py`.

### 4.1 Appearance — `xp_appear`

| Minutes | Points |
| ---: | ---: |
| `m ≤ 0` | 0 (and all other terms 0) |
| `0 < m < 60` | 1 |
| `m ≥ 60` | 2 |

No uncertainty. On historical screens with minutes > 60 pool, this is exactly 2 for every row.

### 4.2 Goals — `xp_goals`

```
xp_goals = (sG × α_G × xG) × GOAL_POINTS[pos]
```

| Pos | GOAL_POINTS |
| --- | ---: |
| GKP | 6 |
| DEF | 6 |
| MID | 5 |
| FWD | 4 |

Share is **within the team-side player table** passed to `score_team_side` (not “all squad”). Players with `rG = 0` get `xp_goals = 0` until they score in prior appearances.

### 4.3 Assists — `xp_assists`

```
xp_assists = (sA × α_A × xG) × 3
```

Same for every position. Independent assist share `sA` from `rA`.

### 4.4 Clean sheet — `xp_cs`

```
xp_cs = 1{m ≥ 60} × xCS × CS_POINTS[pos]
```

| Pos | CS_POINTS |
| --- | ---: |
| GKP | 4 |
| DEF | 4 |
| MID | 1 |
| FWD | 0 |

Same `xCS` for every teammate who clears 60′. **No player share.** FWD term is always 0.

### 4.5 Goals conceded — `xp_gc`

```
xp_gc = 1{m > 0} × (−1) × xGC     for GKP, DEF
xp_gc = 0                          for MID, FWD
```

Gate is **any appearance** (`m > 0`), not 60′. Same `xGC` for every playing GKP/DEF on the side. `xGC = E[⌊GA/2⌋]` from `M`.

### 4.6 Defensive contribution — `xp_defcon`

**Prior** (`history_priors_for_xp`), on starts with `minutes ≥ 60` before GW `t`:

```
μ = expanding mean of defensive_contribution
σ = expanding sample SD   (floor SIG_FLOOR = 0.5)
n = number of prior ≥60′ apps
δ = μ − τ
```

| Pos | Threshold τ | Eligible count | Pts if hit |
| --- | ---: | --- | ---: |
| DEF | 10 | CBIT | 2 |
| MID | 12 | CBIRT | 2 |
| FWD | 12 | CBIRT | 2 |
| GKP | — | — | **0** (τ treated as 0 → term off) |

**Expectation** (normal tail + continuity correction), only if `m ≥ 60`, `τ > 0`, `n > 0`:

```
P(hit) = 1 − Φ( (−0.5 − δ) / σ )
xp_defcon = 2 × P(hit)
SE = 2 · φ((−0.5 − δ)/σ) / √n
```

Not FDR-driven and not λ_A / xGC-driven.

### 4.7 Bonus — `xp_bonus`

**Prior** on appearances (`minutes > 0`), GW `< t`:

```
μ = expanding mean of bonus
SE = σ / √n
xp_bonus = μ    if m > 0 else 0
```

Level term only (own rate). No BPS residual, decile map, or fixture top-3 MC inside this score.

### 4.8 Cards — `xp_cards`

On the same appearance set, card points `−1·yellow − 3·red`:

```
xp_cards = expanding mean of card_pts    if m > 0 else 0
```

(Negative level.)

### 4.9 Saves — `xp_saves` (GKP only)

**Prior:** on appearances (`minutes > 0`), map raw saves through the official step, then expand:

```
y = ⌊saves / 3⌋ = save_points(saves)     # FPL stepped award
μ_pts = expanding mean of y              # = Σ k · p̂_k  (empirical multinomial)
SE_pts = σ_y / √n
```

**Assembler:**

```
xp_saves = μ_pts × m / 90          # GKP and m > 0 only
Δ(xp_saves) via weight_and_delta(μ_pts, SE_pts, m, σ_m)
```

MID/DEF/FWD → 0. Not opponent- or xGC-conditioned. Linear `E[saves]/3` is **out**; stepped multinomial is locked (trial: `data/processed/gk_saves_multinomial_trial_2025_26.csv`).

---

## 5. Position matrix (what each position can score)

| Component | GKP | DEF | MID | FWD |
| --- | :---: | :---: | :---: | :---: |
| Appearance | ✓ | ✓ | ✓ | ✓ |
| Goals (pts/goal) | 6 | 6 | 5 | 4 |
| Assists (×3) | ✓ | ✓ | ✓ | ✓ |
| CS | ×4 | ×4 | ×1 | **0** |
| GC (−xGC) | ✓ | ✓ | **0** | **0** |
| DefCon | **0** | τ=10 | τ=12 | τ=12 |
| Bonus prior | ✓ | ✓ | ✓ | ✓ |
| Cards prior | ✓ | ✓ | ✓ | ✓ |
| Saves (⌊s/3⌋) | ✓ | **0** | **0** | **0** |

---

## 6. Priors and as-of rules

| Object | History filter | Function |
| --- | --- | --- |
| `rG`, `rA` | `minutes > 0`, GW `< t` | `player_rates_from_history` |
| bonus, cards, save_pts | `minutes > 0`, GW `< t` | `history_priors_for_xp` |
| DefCon μ, σ, n | `minutes ≥ 60`, GW `< t` | `history_priors_for_xp` |
| α_G, α_A (screen) | expanding clean sides GW `< t` (pilot/closed) or pooled store | `_asof_alphas` / `load_alphas` |

As-of audit for `score_xp_v2`: rates/priors from before `t` only; minutes on the scored week may be oracle for historical screens (`src/markets/score_xp_v2_asof.py`).

---

## 7. Uncertainty (`xp_se`)

Independent component SEs in quadrature (conservative). Appearance omitted.

| Component | SE source |
| --- | --- |
| Goals / assists | Share SE + α relative SE on `E`, × points |
| CS / GC | `ΔxCS` / `ΔxGC` × position scale (0 if not passed) |
| DefCon | Delta method on residual (see §4.6) |
| Bonus / cards | `SE = σ/√n` of prior |
| Saves | `weight_and_delta` on μ_pts |

**Important:** `xp_se` is narrow relative to point residuals. Do not treat `xp ± xp_se` as a predictive interval for `total_points`. Log-score screens use the point mean only.

---

## 8. Evaluation (what has been measured)

### 8.1 Gated closed-season screen vs `score_xp`

Protocol: `experiments/protocol.json` → `score_xp_v2_screen`.  
Report: `reports/score_xp_v2_gate.md`.

- Pool: minutes > 60, FD opening AH+OU, join to closed `score_xp` (`n_prior ≥ 3`), GW 5–38.
- Minutes oracle; σ_m = 0; Gaussian log-score σ = 3.
- **Δlog (`score_xp_v2` − `score_xp`):** **+0.0531** [+0.0453, +0.0609] over four seasons (27 568 player-fixtures).
- Interval stays above zero. **Does not** replace published `score_xp`.

| Season | Mean Δlog | n | MAE v2 | MAE score_xp |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | +0.0768 | 6864 | 2.285 | 2.272 |
| 2023-24 | +0.0435 | 6956 | 2.320 | 2.294 |
| 2024-25 | +0.0306 | 6892 | 2.254 | 2.239 |
| 2025-26 | +0.0622 | 6856 | 2.364 | 2.375 |

### 8.2 Four-season positional performance (closed file)

Pool as gated screen. Bias = mean(`score_xp_v2`) − mean(`total_points`).

| Pos | n | Bias | MAE v2 | MAE score_xp | Spearman(v2, pts) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GKP | 2683 | **−0.705** | 2.017 | 2.264 | 0.160 |
| DEF | 10546 | −0.072 | 2.349 | 2.383 | 0.282 |
| MID | 11658 | −0.048 | 2.197 | 2.109 | 0.250 |
| FWD | 2681 | +0.015 | 2.893 | 2.787 | 0.178 |

GKP low bias is stable across seasons (−0.61 to −0.77). That is largely **omitted / late saves** in the closed rebuild (see §8.4). DEF/MID nearly unbiased. FWD slightly high in 2024-25 only.

### 8.3 Component decomposition — 2025/26 pilot panel

Source: `data/processed/numagent_xp_pilot_2025_26.csv` (minutes > 60, FD openings).  
**Note:** that CSV was written **before** the multinomial saves lock, so predicted `xp_saves = 0` in the table below. Actual save points are still shown. Current code emits non-zero `xp_saves` for GKP.

**Overall:** mean xp 3.69 vs pts 3.83 (bias −0.14); MAE 2.38.

**Component mass and error (all positions):**

| Component | Mean pred | Mean act | Bias | MAE | ≈% of pred xp | ≈% of component SSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| appear | 2.000 | 2.000 | 0 | 0 | 54% | 0 |
| goals | 0.547 | 0.539 | +0.01 | 0.86 | 15% | **42%** |
| assists | 0.323 | 0.308 | +0.02 | 0.54 | 9% | 15% |
| cs | 0.612 | 0.633 | −0.02 | 0.86 | 17% | 21% |
| gc | −0.241 | −0.209 | −0.03 | 0.24 | −7% | 2% |
| defcon | 0.352 | 0.369 | −0.02 | 0.46 | 10% | 8% |
| bonus | 0.242 | 0.297 | −0.06 | 0.45 | 7% | 9% |
| cards | −0.144 | −0.156 | +0.01 | 0.25 | −4% | 2% |
| saves | 0.000† | 0.058 | −0.06 | 0.06 | 0† | 1 |

† Pre-saves pilot CSV. Locked assembler uses multinomial `⌊saves/3⌋` prior.

Goals dominate residual **variance** with almost no mean bias (classic blank/haul). See goals diagnosis in `REPORT_STATE` milestone `score_xp_v2-goals-gap-diagnosis`.

### 8.4 Positional component means — 2025/26 pilot

Predicted means from the pilot assembler; actuals from FPL scoring rules on the same rows.

**GKP (n=735)** — mean xp 2.71 vs pts 3.35 (bias **−0.64** without saves)

| Term | Pred | Act | Bias |
| --- | ---: | ---: | ---: |
| appear | 2.00 | 2.00 | 0 |
| goals | 0.00 | 0.00 | 0 |
| assists | 0.03 | 0.02 | +0.01 |
| cs | 1.04 | 1.00 | +0.05 |
| gc | −0.49 | −0.46 | −0.04 |
| bonus | 0.21 | 0.21 | +0.01 |
| cards | −0.08 | −0.07 | −0.01 |
| saves | 0.00† | **0.59** | **−0.59** |

Almost the entire GKP level gap is saves. Multinomial lock targets that hole (~0.2–0.9 xp by keeper).

**DEF (n=2922)** — mean xp 3.58 vs pts 3.69 (bias −0.11)

| Term | Pred | Act | Bias |
| --- | ---: | ---: | ---: |
| appear | 2.00 | 2.00 | 0 |
| goals | 0.25 | 0.26 | −0.00 |
| assists | 0.23 | 0.22 | +0.01 |
| cs | 1.03 | 1.06 | −0.03 |
| gc | −0.50 | −0.42 | −0.07 |
| defcon | 0.53 | 0.55 | −0.01 |
| bonus | 0.19 | 0.22 | −0.03 |
| cards | −0.17 | −0.17 | +0.01 |

CS + DefCon are large positive mass; GC slightly too harsh (market xGC a bit high).

**MID (n=3103)** — mean xp 3.91 vs pts 3.99 (bias −0.08)

| Term | Pred | Act | Bias |
| --- | ---: | ---: | ---: |
| appear | 2.00 | 2.00 | 0 |
| goals | 0.73 | 0.71 | +0.01 |
| assists | 0.49 | 0.45 | +0.03 |
| cs | 0.26 | 0.30 | −0.03 |
| defcon | 0.34 | 0.37 | −0.03 |
| bonus | 0.24 | 0.33 | −0.09 |
| cards | −0.15 | −0.17 | +0.02 |

Attack level fine; bonus slightly low; DefCon slightly low.

**FWD (n=750)** — mean xp 4.18 vs pts 4.21 (bias −0.04)

| Term | Pred | Act | Bias |
| --- | ---: | ---: | ---: |
| appear | 2.00 | 2.00 | 0 |
| goals | 1.49 | 1.46 | +0.04 |
| assists | 0.31 | 0.34 | −0.03 |
| defcon | 0.02 | 0.02 | −0.01 |
| bonus | 0.46 | 0.54 | −0.08 |
| cards | −0.10 | −0.11 | +0.01 |

CS/GC/saves are structurally 0. Goals dominate predicted attack mass (~36% of FWD xp).

### 8.5 Predicted mix by position (% of mean xp)

| | GKP | DEF | MID | FWD |
| --- | ---: | ---: | ---: | ---: |
| appear | 74% | 56% | 51% | 48% |
| goals | 0% | 7% | 19% | 36% |
| assists | 1% | 6% | 12% | 8% |
| cs | 38% | 29% | 7% | 0% |
| gc | −18% | −14% | 0% | 0% |
| defcon | 0% | 15% | 9% | 0% |
| bonus | 8% | 5% | 6% | 11% |
| cards | −3% | −5% | −4% | −2% |
| saves† | 0%† | 0% | 0% | 0% |

† After saves lock, expect GKP saves roughly **+15–25%** of GKP xp for high-volume keepers (replacing part of the appearance-dominated share).

---

## 9. Explicitly out of `score_xp_v2`

- Soft shrink of `r` toward a position prior  
- Season-unconditional goal shares (injury blanks in the rate)  
- Attack allocation by `minutes / team_minutes`  
- Linear `E[saves]/3`  
- Opponent / xGC-driven saves  
- FDR- or λ_A-driven DefCon  
- BPS-residual percentiles / decile maps / fixture top-3 MC inside xBonus  
- Own goals, pens  
- Replacing published `score_xp` on climbs without a separate decision  
- Fitting anything on frozen 2026/27 holdout  

**Not out on live (wanted):** BTTS in the scoreline fit when priced; OU totals other than 2.5; AH lines beyond the usual ±0.5/±1 band. Historical FD screens stay AH+OU2.5, BTTS weight 0.

---

## 10. Code map

| Piece | Path |
| --- | --- |
| Assembler | `src/markets/numagent_xp.py` |
| Scoreline / moments | `src/markets/scoreline.py` |
| α store | `src/markets/fpl_score_rates.py`, `data/processed/fpl_goal_assist_rates.json` |
| FPL constants | `src/rules/fpl_2026.py` |
| 2025/26 pilot | `src/markets/numagent_xp_pilot.py` |
| Four-season closed | `src/markets/numagent_xp_closed.py` |
| Gate writer | `src/markets/score_xp_v2_gate.py` |
| As-of audit | `src/markets/score_xp_v2_asof.py` |
| Protocol screen | `experiments/protocol.json` → `score_xp_v2_screen` |
| Unit tests | `tests/test_numagent_xp.py`, `tests/test_numagent_xp_uncertainty.py` |

### Artefacts

- `data/processed/score_xp_v2_closed.csv` / `_gw.csv` / `score_xp_v2_gate_intervals.json`  
- `data/processed/numagent_xp_pilot_2025_26.csv`  
- `data/plots/score_xp_v2_closed_dlog.png`  
- `data/processed/gk_saves_multinomial_trial_2025_26.csv`  
- `data/plots/gk_saves_multinomial_trial_2025_26.png`  

---

## 11. How to re-read this later

1. Total identity §3 and position matrix §5 — what can fire.  
2. Component §4 — exact formula and gates.  
3. §8 — whether the term is calibrated (level) or only noisy (goals/CS variance).  
4. Diff the assembler against this doc if behaviour looks wrong; the code is source of truth if they diverge — update this report.

*Generated from the live assembler and closed/pilot panels as of 2026-10-14.*
