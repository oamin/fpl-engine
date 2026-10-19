# score_xp_v2 — engine primary scorer

Locked: 2026-10-13. **Promoted as engine primary 2026-10-19** (`protocol.primary_scorer`). **`score_xp` is superseded.** This is the expected-points construction from market scorelines and appearance rates.

**Stack index:** [`LOCKED.md`](LOCKED.md) · live entry: `python -m src.live.numagent_live_pick` · strategiser default: L4 priced γ=1.

**Comprehensive reference** (every component as implemented, position matrix, closed/pilot diagnostics): [`reports/score_xp_v2.md`](score_xp_v2.md).

## Pipeline

1. **Scoreline matrix M** — Dixon–Coles fit to fair market targets. Parameters (λ_H, λ_A, ρ).
   - **Historical / FD screens:** AH + **OU 2.5**, BTTS weight **0** (BTTS not on free FD openings).
   - **Live (PI):** include **BTTS when available**; support **OU lines other than 2.5** (3.0, 3.5, …) and whatever **AH line** the book quotes (±3, ±3.5, …). Fall back to BTTS weight 0 only if missing.

2. **Team moments from M**
   - **xG** = E[goals for]
   - **xCS** = P(goals against = 0)
   - **xGC** = E[⌊GA / 2⌋] (GC *events*, not ⌊E[GA] / 2⌋)

3. **FPL award scalings** (match goal → FPL player goal / assist)  
   File: `data/processed/fpl_goal_assist_rates.json` · helper `src/markets/fpl_score_rates.py`

   | Scaling | Role | Prior 2025/26 (clean) |
   | --- | --- | ---: |
   | α_G | FPL player goals per scoreline match goal | **0.9615 ± 0.0061** |
   | α_A | FPL assists per scoreline match goal | **0.9027 ± 0.0095** |

   Team pots from the matrix:

   ```
   G_team = α_G × xG_M
   A_team = α_A × xG_M
   ```

   **Live update (required, PI 2026-10-16):** α_G and α_A start from the **2025/26 prior** and **evolve continuously** with 2026/27. After each completed live GW:

   ```
   α = (counts_prior + counts_live_to_date) / (match_goals_prior + match_goals_live_to_date)
   ```

   Refresh: `python -m src.markets.fpl_score_rates --refresh-live` (optional `--before-gw N` for as-of). Match goals from FD `E0_2627`; player G/A from `player_gw_2026_27`. Drop sides with player goals > match goals. Write `live` + `pooled` to the JSON. Do not freeze 0.96/0.90 for the year. **Shares stay same-season only** — α is the only prior-season blend.

4. **Player rates** (appearance-only; early-season M0 — promoted 2026-10-16)
   - **Same season only** (PI 2026-10-16): history is GW `< t` within the season being scored. **No prior-season fill** for `rG`/`rA` → shares. Helper: `asof_share_history` / `assert_single_season_rate_history` in `numagent_xp.py`. Early-season empty history → rates 0 until appearances accrue. (α_G/α_A may still pool a prior for the **team pot scale**; that is not share mass.)
   - Rows with minutes > 0 only (injury blanks excluded).
   - Minutes pseudo-count + zero-event SE floor (`protocol.score_xp_v2_rate_m0`):
     - `rG_i = G_i / (M_i + M0)`,   `ΔrG_i = max(√G_i, c) / (M_i + M0)` with `M0=270`, `c=0.25`
     - same for assists (`rA` / `ΔrA`)
     - Not soft shrink toward a position prior; not prior-season fill.
   - Gated vs raw MLE rates: Δ **+0.0037** [+0.0006, +0.0071] (`reports/score_xp_v2_rate_m0.md`).

5. **Minutes oracle** — **default = rolling 3-GW mean** (zeros included); tags **override** only for clear context.
   - Compile path: `compile_player_xmi` in `src/live/news_packets.py` → `xmi_compiled`, `xmi_sigma`.
   - Live sheet may carry optional column `xmi_sigma` (`src/live/xmi.py`).
   - Default `m_i`: mean of last 3 GW minutes; if empty series, appearance prior. Team-news LLM emits contextual tags and optional **expected lineups**; **`none` replaces ask** (rolling stays).
   - Override tags: `injured`, `transferred`, `benched`, `50/50`. Hard FPL flags (`s`/`u`/chance 0) → 0.
   - Expected lineup (per club): `xi` (11 ids, one GKP) + optional `bench`. Placement → starter / bench / out minutes. `injured`/`transferred` beat lineup; lineup beats soft tags and rolling.
   - Spread `σ_m` (3-GW window, zeros included unless noted):
     - `xmi = 0` (out) → `σ_m = 0`
     - `50/50` → mixture SD of {0, prior} (= half the base)
     - `firm_starter` (diagnostic only; xmi still rolling) → SD of ≥60′ apps if ≥2, else full-window SD
     - `rolling avg` / `benched` / other → sample SD of the last 3 GW minutes
   - Into weights (independent rate and minutes errors):

   ```
   Δw_i = sqrt( (m_i/90 × Δr_i)² + (r_i/90 × σ_m,i)² )
   ```

6. **Weights and shares**
   - `w_i = r_i × m_i / 90`,   `Δw_i` as in §5 (not rate-only)
   - `W = sum_j w_j`,   `ΔW = sqrt(sum_j (Δw_j)²)` (players independent)
   - `s_i = w_i / W`
   - Share SE (do **not** treat `w_i` and `W` as independent):

   ```
   Δs_i = (1 / W) × sqrt( (1 − s_i)² (Δw_i)² + s_i² × sum_{k≠i} (Δw_k)² )
   ```

6b. **Share-SE shrink** (promoted 2026-10-17; `protocol.score_xp_v2_share_se_pos_prior`)

   Fragile compositional shares (sparse early-season scorers, e.g. CB at 37%
   of club goals) shrink toward a **position×minutes** prior:

   - `s0_i ∝ m_i · w(pos_i)` (locked closed-season per-90 weights; MID=1;
     GKP goals 0; DEF goals ≈0.255)
   - `τ² = median(Δs²)`; `K = τ²/(τ²+Δs²)`; `s̃ = Ks+(1−K)s0`; renormalise

   Default on in `score_team_side`. Gated win vs raw: Δ **+0.0195**
   [+0.0158, +0.0234] (`reports/score_xp_v2_share_se_pos_prior.md`).
   All-minutes and scorers-only priors parked.

7. **Attack expectations**
   - `E[G_i] = sG_i × G_team`  (using shrunk `s̃` when share-SE shrink is on)
   - `E[A_i] = sA_i × A_team`
   - With pot uncertainty (relative quadrature):

   ```
   ΔE / E = sqrt( (Δs / s)² + (ΔG_team / G_team)² )
   ```

7b. **Live props overlay** (PI 2026-10-16; **lift-only**) — `prop_shares.rescale_goal_shares`

   - Base live μ (`mu_post` ≡ `score_xp_v2`) from AH/OU/**BTTS** → M → `G_team = α_G × xG_M`.
   - Anytime markets quote **P(G≥1)**, not E[G]. Poisson would force `E[G]=λ=−log(1−p)`, which can **cut** a star whose baseline share already has E[G]>1. So props are a **lift**, not a hard replace:
     target `max(baseline_i, λ(p_i))` for covered `m>0` players.
   - Extra mass is taken from **uncovered** teammates first (props win on the margin without shrinking a Haaland baseline). Only if that is not enough and Σ would exceed `G_team`, scale everyone.
   - Assists: no prop overlay (US books thin).

8. **Defence from M** (simple, fixed position rates — no player shares)

### Clean sheet (CS)

From M (home side; flip index if away):

```
xCS = P(GA = 0) = sum_i M[i, 0]
```

FPL awards a CS only if the player plays **≥ 60 minutes**. Points by position (`CS_POINTS` in `fpl_2026.py`):

| Pos | CS pts |
| --- | ---: |
| GKP | 4 |
| DEF | 4 |
| MID | 1 |
| FWD | 0 |

Oracle gate (binary; no smooth p60):

```
E[xp_CS,i] = 1{m_i ≥ 60} × xCS × CS_PTS(pos_i)
```

Same `xCS` for every teammate who clears 60′. No share. FWD term is always 0.

Uncertainty:

```
Δ(xp_CS,i) = 1{m_i ≥ 60} × CS_PTS(pos_i) × ΔxCS
```

when `xCS` has an SE from M.

### Goals conceded (GC)

```
xGC = E[⌊GA / 2⌋] = sum_{i,j} M[i,j] × ⌊j / 2⌋
```

| Who | Gate | Expectation |
| --- | --- | --- |
| GKP, DEF | `m_i > 0` (any appearance; **not** 60′) | `−xGC` |
| MID, FWD | — | **0** |

Same `xGC` for every playing GKP/DEF. `Δ(xp_GC) = ΔxGC` (rate −1).

### Defensive contribution (DefCon)

Source count: FPL `defensive_contribution` on starts with minutes ≥ 60 (appearance rows only).

| Pos | Count | Threshold `τ` | Pts |
| --- | --- | ---: | ---: |
| DEF | CBIT | 10 | 2 |
| MID, FWD | CBIRT | 12 | 2 |
| GKP | — | — | 0 |

**Residual to threshold** (locked state): on prior starts (≥60′), **expanding** mean and sample SD of `defensive_contribution`:

```
μ_i = expanding mean
σ_i = expanding sample SD   (floor σ_min = 0.5)
δ_i = μ_i − τ_i
```

`δ_i` is the useful xDefCon feature — how far above/below the line — with `σ_i` the uncertainty. Not FDR or λ_A.

**xDefCon** via normal tail on the residual (continuity correction):

```
P(hit_i) = P(X ≥ τ) ≈ 1 − Φ( (−0.5 − δ_i) / σ_i )
E[xp_DefCon,i] = 1{m_i ≥ 60} × 2 × P(hit_i)
```

Uncertainty on the expectation (delta method on `δ`, `SE_δ = σ/√n`):

```
SE(xp_DefCon) = 2 · φ((−0.5 − δ)/σ) / √n
```

Level diagnostic (not a next-GW claim): xDefCon tracks expanding mean DefCon pts (r≈0.88 on probe set); next-GW r≈0.13. Plots: `defcon_residual_expanding_selected_2025_26.png`, `xdefcon_vs_expanding_pts_selected_2025_26.png`, `xdefcon_level_vs_nextgw_caveat_2025_26.png`.

### Bonus (xBonus) — prior μ ± SE

FPL bonus is 3/2/1 from fixture BPS rank. Same-GW prediction is weak; the useful object is the player’s **own bonus rate**.

On appearances with `minutes > 0`, using only **completed** GWs before the forecast week (leakage-free **prior**):

```
μ_i  = expanding mean of bonus_i     (prior; no current-GW row)
σ_i  = expanding sample SD of bonus_i
n_i  = number of prior appearances
SE_i = σ_i / √n_i
```

```
E[xp_bonus,i] = μ_i
Δ(xp_bonus,i) = SE_i
```

Report the interval clipped at the natural floor (bonus ≥ 0):

```
[max(0, μ_i − SE_i), μ_i + SE_i]
```

`SE_i` itself is unchanged; only the displayed / propagated lower edge is clipped. No other gates, maps, percentiles, or FDR. Minutes already decide whether the player is in the side (`m_i`). After each live GW completes, append that bonus and refresh `μ` / `SE` for the next deadline.

Plot: `data/plots/xbonus_expanding_mean_se_2025_26.png` (solid = inclusive diagnostic; **dashed prior μ** is the locked xP term).

2025/26: same-GW r≈0.10; player-season mean r≈0.73 — **level**, not a week-picker. BPS-residual / decile / MC-rank machinery stays out of xP.

### Cards (xCards) — prior μ ± SE

Same recipe as bonus on appearances (`minutes > 0`), using card points `−1·Y − 3·R`:

```
E[xp_cards,i] = μ_i^{cards}     (prior expanding mean)
Δ(xp_cards,i) = SE_i^{cards}
```

Clip the reported upper edge at 0 (cards ≤ 0): `[μ − SE, min(0, μ + SE)]`. Level term only (~4.5–5.7% of season point mass).

9. **xP_i** (assembler: `src/markets/numagent_xp.py`)

### GKP saves

On prior appearances (minutes > 0), expand the empirical multinomial of stepped FPL save points `y = ⌊saves / 3⌋` (via `save_points`):

```
μ_pts = expanding mean of y          (= Σ k · p̂_k)
SE_pts = σ_y / √n                    (SE of that mean)
xp_saves = μ_pts × m / 90
Δ(xp_saves) = √( (m/90 · SE_pts)² + (μ_pts/90 · σ_m)² )
```

GKP only; 0 for other positions and for `m ≤ 0`. Level prior (opponent/xGC not in this term). `Δ(xp_saves)` enters total `xp_se` quadrature as stability of the mean, not a predictive band for the discrete step. Trial vs linear `/3` on 2025/26 nailed GKs: `data/processed/gk_saves_multinomial_trial_2025_26.csv`.

```
xP = xp_appear + xp_goals + xp_assists + xp_cs + xp_gc + xp_defcon + xp_bonus + xp_cards + xp_saves
```

Appearance from the minutes oracle: `1{m_i > 0} + 1{m_i ≥ 60}`. If `m_i ≤ 0`, all terms are 0. Own goals and pens stay at 0. `score_xp_v2` is the engine primary scorer; `score_xp` is superseded.

## Explicitly out

- Soft Bayesian shrink of `r_i` toward a position prior (unless a later gated comparison requires it).
- Season-unconditional goal shares that dilute injury blanks into the rate.
- Allocating attack by `minutes / team_minutes`.
- Matching historical FD openings to Odds API BTTS (not available on free historical). **Live** BTTS is wanted when priced — see scoreline §1.
- Extending superseded `score_xp` climbs for new claims (replay-only).
- Linear `E[saves]/3` for GKP saves (stepped multinomial `E[⌊saves/3⌋]` is locked).
- Opponent / xGC-driven saves rates (level expanding multinomial is the locked path for now).
- Driving DefCon from FDR or λ_A / xGC (expanding residual `δ=μ−τ` with `σ` is the locked path).
- BPS-residual percentiles, decile/`f(δ)` maps, or fixture top-3 MC inside the locked **xBonus** term (expanding own bonus mean only).

## Reference trials

- FD AH+OU season moments: `reports/scoreline_fd_moments_2025_26.md`
- Rates store: `src/markets/fpl_score_rates.py`
- Scoreline fit: `src/markets/scoreline.py`
- Rotation demo (Saka / Havertz / Gyökeres): chat lock 2026-10-13
