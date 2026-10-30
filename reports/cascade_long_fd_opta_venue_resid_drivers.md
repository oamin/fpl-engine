# Per-season drivers of home/away Opta−FD residual

Blind Opta vs FD · GW8–38 · four closed seasons.

## Verdict

One coherent mechanism across seasons:

1. **Blind Opta is almost venue-blind** — mean λ_H ≈ mean λ_A every season (~1.43–1.52).
2. **FD prices a large home advantage** (HA ≈ +0.30 to +0.32 goals).
3. Therefore **resid_H ≈ Opta_level − FD_H < 0** and **resid_A ≈ Opta_level − FD_A > 0** whenever the book’s away λ is soft.

Season differences are mostly **how hot/cold Opta’s place level is vs actual**, and **whether FD away is soft** — not a different home/away mechanism each year.

## Season × venue anatomy

| season | HA act | HA FD | HA Opta | resid H | resid A | Opta−act | FD−act |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | +0.328 | +0.304 | -0.000 | -0.068 | +0.237 | +0.083 | -0.002 |
| 2023-24 | +0.352 | +0.318 | +0.012 | -0.320 | -0.014 | -0.247 | -0.080 |
| 2024-25 | +0.087 | +0.327 | -0.010 | -0.251 | +0.086 | -0.002 | +0.080 |
| 2025-26 | +0.259 | +0.322 | -0.009 | -0.099 | +0.232 | +0.121 | +0.054 |

## Decomposition (`resid = Opta_bias − FD_bias`)

| season | ha | resid | Opta−act | FD−act |
| --- | --- | ---: | ---: | ---: |
| 2022-23 | H | -0.068 | -0.082 | -0.014 |
| 2022-23 | A | +0.237 | +0.247 | +0.010 |
| 2023-24 | H | -0.320 | -0.417 | -0.097 |
| 2023-24 | A | -0.014 | -0.077 | -0.063 |
| 2024-25 | H | -0.251 | -0.051 | +0.200 |
| 2024-25 | A | +0.086 | +0.047 | -0.039 |
| 2025-26 | H | -0.099 | -0.013 | +0.086 |
| 2025-26 | A | +0.232 | +0.255 | +0.023 |

## Per-season reading

### 2022-23

Opta H/A nearly flat (1.50/1.50); FD HA=+0.30, act HA=+0.33. Mild negative home resid (-0.07): FD home 1.57 vs Opta 1.50. Large positive away resid (+0.24): Opta away 1.50 vs soft FD away 1.26. Season-level: Opta hot vs actual (+0.08) — place prior low vs a cold year. FD vs act: H -0.01 / A +0.01; Opta vs act: H -0.08 / A +0.25.

### 2023-24

Opta H/A nearly flat (1.43/1.42); FD HA=+0.32, act HA=+0.35. Large negative home resid (-0.32): book prices a strong home λ (1.75) while Opta stays ~1.43. Away resid near/below 0 (-0.01): FD away not soft (FD_A=1.43 ≈ Opta_A=1.42). Season-level: Opta cold vs actual (-0.25) — LOSO place prior high vs a high-scoring year. FD vs act: H -0.10 / A -0.06; Opta vs act: H -0.42 / A -0.08.

### 2024-25

Opta H/A nearly flat (1.48/1.49); FD HA=+0.33, act HA=+0.09. Large negative home resid (-0.25): book prices a strong home λ (1.73) while Opta stays ~1.48. Mild positive away resid (+0.09). FD vs act: H +0.20 / A -0.04; Opta vs act: H -0.05 / A +0.05.

### 2025-26

Opta H/A nearly flat (1.51/1.52); FD HA=+0.32, act HA=+0.26. Mild negative home resid (-0.10): FD home 1.61 vs Opta 1.51. Large positive away resid (+0.23): Opta away 1.52 vs soft FD away 1.29. Season-level: Opta hot vs actual (+0.12) — place prior low vs a cold year. FD vs act: H +0.09 / A +0.02; Opta vs act: H -0.01 / A +0.25.

## Cross-season compression

Because Opta H/A means are flat, venue residuals are essentially `Opta_season_level − FD_venue_mean`:

| season | resid H | level−FD_H | resid A | level−FD_A |
| --- | ---: | ---: | ---: | ---: |
| 2022-23 | -0.068 | -0.067 | +0.237 | +0.237 |
| 2023-24 | -0.320 | -0.326 | -0.014 | -0.008 |
| 2024-25 | -0.251 | -0.246 | +0.086 | +0.081 |
| 2025-26 | -0.099 | -0.095 | +0.232 | +0.227 |

### Why 2023-24 away is the odd one

Only season where away resid ≈ 0: FD away mean (**1.43**) matched Opta away (**1.42**). Home resid is still deeply negative because FD home was very high (**1.75**) in a high-scoring home season (act H **1.85**), while Opta stayed ~**1.43**. So 2023-24 is **Opta cold + book HA large**, not a different away mechanism.

## Implication

A fixed `r_H, r_A` scale fails OOS because **FD venue means and Opta season level move year to year**; the *shape* (Opta flat vs book HA) is stable, the *gap size* is not.

