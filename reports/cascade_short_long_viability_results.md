# Cascade short→long viability — overnight results

Batch finished. Locked stack: centreline `M` → `k ∼ M` (MARKET_BAND=0.0).

## A — Integrity

- Passed: **True**
- As-of audit: **True**
- No λ jitter: **True**
- k∈supp(M): **True**
- Routing/props: **True**
- MILP units: **True**

## B — Likelihood (player-GW log)

- Window GW8–28 · N_MC=6 · σ=3.0
- Verdicts: `{'short_gen_minus_short_mean': 'clearly_loses', 'long_gen_minus_long_mean': 'clearly_loses', 'short_gen_minus_long_gen': 'does_not_clearly_lose'}`
- GW pairs: B1=84 B2=84 B3=84

- `short_gen_minus_short_mean`: Δ mean=-0.0892 [-0.1002, -0.0780] · **clearly_loses**
- `long_gen_minus_long_mean`: Δ mean=-0.0900 [-0.1005, -0.0793] · **clearly_loses**
- `short_gen_minus_long_gen`: Δ mean=-0.0005 [-0.0141, +0.0127] · **does_not_clearly_lose**

- Gated markdown written: **False**

## C — FT + captain (diagnostic)

- Smoke passed: **True**
- Squads: 72 · mean transfers 3.8833333333333333 · mean hits 7.533333333333333
- Sources in sample: ['cascade_long_opta', 'cascade_short_priced']
- Scoring: GW8 cascade_short priced centreline M→k∼M (props matched=330; no λ jitter); GW≥t+2 Opta place→GF/GA±MAE→M→k∼M; generative lineup→attack→CS/GC; soft priors from score_xp_v2 helpers; same-season as-of rates/priors/rolling m frozen at decision GW

Top FT freq:

- Groß (brighton): freq=0.931 xi=0.9305555555555556 cap=0.1388888888888889
- Tzolakis (hull): freq=0.917 xi=0.1388888888888889 cap=0.0
- Tarkowski (everton): freq=0.889 xi=0.8611111111111112 cap=0.20833333333333334
- Dasilva (coventry): freq=0.847 xi=0.6805555555555556 cap=0.125
- Hall (newcastle): freq=0.847 xi=0.75 cap=0.1111111111111111
- Van Hecke (spurs): freq=0.750 xi=0.6388888888888888 cap=0.1111111111111111
- Wirtz (liverpool): freq=0.736 xi=0.6805555555555556 cap=0.05555555555555555
- Saka (arsenal): freq=0.556 xi=0.5277777777777778 cap=0.06944444444444445
- Hume (sunderland): freq=0.556 xi=0.0 cap=0.0
- N.Jackson (aston villa): freq=0.514 xi=0.4027777777777778 cap=0.013888888888888888

## D — Smoke

- Passed: **True**

## Morning decision aid

| Arm | Result |
| --- | --- |
| A integrity | True |
| B1 short gen−mean | clearly_loses |
| B2 long gen−mean | clearly_loses |
| B3 short_gen−long_gen | does_not_clearly_lose |
| C FT+cap smoke | True |
| D regression | True |

Artifacts: `data/processed/cascade_viability/`

Reproduce:
```bash
PYTHONPATH=. python3 -m src.eval.cascade_viability_batch
```

