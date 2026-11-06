# Mean t:t+1 vs short generative

Protocol: `cascade_short_mean_tt1_vs_gen` · N_MC=20 · σ=3.0 · decision GW8–27 · seed=7

Treatment: generative MC-mean on FD centreline fits for GW(t) and GW(t+1).  
Baseline: `score_team_side` on the same fits.  
As-of rates/minutes frozen at decision GW t for both weeks.

- As-of audit: **True**
- Decision-GW pairs: 80
- Δ mean (gen − mean): **-0.0539** [-0.0597, -0.0484]
- Verdict: **clearly_loses**
- Wall: 1858.0s

Per season n_gws: 2022-23=20, 2023-24=20, 2024-25=20, 2025-26=20

Artifacts: `data/processed/cascade_mean_tt1/`

Reproduce:
```bash
PYTHONPATH=. python3 -m src.eval.cascade_short_mean_tt1 --n-mc 20
```
