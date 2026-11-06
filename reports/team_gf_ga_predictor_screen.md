# Team-level predictors of goals for / against

Leakage-free screen across **2022/23–2025/26** (2,800 team–matches with ≥3 prior games; 2,730 with 1X2). Predictors use only matches before the fixture. Market features are shin-style probs from football-data Avg/B365: `attack_strength = p_win + 0.5 p_draw`, `defend_threat = p_lose + 0.5 p_draw`.

Question: what forecasts a club’s **GF** and **GA** well enough to build a **pseudo-match FDR** for long horizons?

## Next match (H=1) — top Spearman ρ

### Goals for

| predictor | ρ | Pearson | OLS R² | MAE |
|---|---:|---:|---:|---:|
| market λ proxy (`2.5·p_over·atk/(atk+def)`) | +0.34 | +0.36 | 0.132 | 0.92 |
| `p_win` | +0.33 | +0.35 | 0.124 | 0.93 |
| `attack_strength` | +0.33 | +0.34 | 0.118 | 0.93 |
| **opp roll3 GA** | +0.33 | +0.33 | 0.107 | 0.93 |
| λ = ½(own exp GF + opp exp GA) | +0.32 | +0.33 | 0.109 | 0.95 |
| opp venue GA | +0.28 | +0.27 | 0.075 | 0.96 |
| DC-style λ (own xG × opp xGC) | +0.25 | +0.27 | 0.072 | 0.97 |
| own exp GF | +0.22 | +0.23 | 0.052 | 0.98 |
| λ from xG blend | +0.23 | +0.20 | 0.042 | 0.99 |
| own GF dispersion | ~0 | — | — | — |

### Goals against

| predictor | ρ | Pearson | OLS R² | MAE |
|---|---:|---:|---:|---:|
| **opp roll3 GF** | +0.37 | +0.40 | 0.159 | 0.92 |
| `defend_threat` | +0.33 | +0.34 | 0.118 | 0.93 |
| `p_win` (negative) | −0.33 | −0.33 | 0.109 | 0.94 |
| λ = ½(own exp GA + opp exp GF) | +0.32 | +0.34 | 0.117 | 0.94 |
| opp venue GF | +0.32 | +0.35 | 0.125 | 0.94 |
| opp exp GF | +0.30 | +0.32 | 0.100 | 0.95 |
| own exp GA | weaker than opponent attack | | | |

## Multivariate (H=1)

| model | R² | MAE |
|---|---:|---:|
| GF ~ `attack_strength` | 0.118 | 0.933 |
| GF ~ λ_avg + `attack_strength` | 0.130 | 0.926 |
| GF ~ own roll3 xG + opp roll3 xGC + `attack_strength` | 0.132 | 0.923 |
| GA ~ opp roll3 GF | 0.159 | 0.915 |
| GA ~ `defend_threat` | 0.118 | 0.932 |
| GA ~ λ_xG + `defend_threat` | 0.122 | 0.930 |

Adding dispersion does nothing. Home flag adds almost nothing once market is present.

## Holdout 2025/26 (fit on 2022–24)

Market alone transfers; raw xG λ does not.

| model | holdout R² | MAE |
|---|---:|---:|
| GF ~ `attack_strength` | **0.080** | 0.88 |
| GF ~ λ_xG | 0.006 | 0.95 |
| GA ~ `defend_threat` | **0.080** | 0.88 |
| GA ~ λ_xG | 0.008 | 0.95 |

## Longer horizons (mean GF / GA over next H)

Spearman rises with H for **own expanding rates** (rate signal). Market stays strong for the current priced week but for H≥5 **own exp GF/GA** is the best single rate prior; opponent form fades because the opponent changes each week.

| H | best GF ρ | best GA ρ |
|---:|---|---|
| 1 | market λ 0.34 | opp roll3 GF 0.37 |
| 3 | market λ 0.37 | defend_threat 0.31 |
| 5 | **own exp GF 0.44** | defend_threat / own exp GA ~0.31 |
| 8 | **own exp GF 0.50** | **own exp GA 0.34** |

## Read for a pseudo-match FDR

Split **team rate** from **fixture difficulty**:

1. **Team baseline (horizon):** expanding mean GF / GA (or xG / xGC). These dominate H=5–8 averages.
2. **This fixture’s FDR (match pot):**
   - **When 1X2 is priced:** `attack_strength` / `defend_threat` (or `p_win`) — best next-match GF/GA predictors and the only ones that hold out cleanly on 2025/26.
   - **When unpriced (true long horizon):** opponent **roll3 or expanding GF** as the threat you face for GA, and opponent **roll3 or expanding GA** as the softness for GF. Opp roll3 GF is the single best GA predictor in-sample (ρ 0.37).
3. **Blend for a match λ:**
   - `λ_GF ≈ ½(own_att + opp_def)` with `own_att = exp GF` (or market attack when present) and `opp_def = opp exp/roll3 GA` (or market defend threat).
   - `λ_GA ≈ ½(own_def + opp_att)` symmetrically.
4. Do **not** lean on dispersion for the mean FDR; it does not forecast GF/GA.

Practical FDR schedule for open weeks: for each future opponent, store that opponent’s current expanding (or roll3) GF and GA; your pseudo-match pot is the blend above. Replace with Betfair/1X2 `attack_strength` when the week is priced.

## Artefacts

- `reports/figures/team_gf_ga_predictor_screen.json`

Reproduce: re-run the screen block in the agent log, or ask for `scripts/screen_team_gf_ga_predictors.py` to be checked in.
