# Lineup-off trial — freeze minutes on short generative

Protocol: `cascade_lineup_off_tt1` · N_MC=20 · σ=3.0 · decision GW8–27 · seed=7

Treatment: generative MC-mean with **`lineup_cov=False`** (as-of mean minutes; scoreline → attack → CS/GC only).  
Baseline: `score_team_side` on the same FD centreline fits.  
Diagnostic: same gen with `lineup_cov=True` (joint historical pairs).

- As-of audit: **True**
- Decision-GW pairs: 80
- Wall: 2661.8s

## Primary — gen_off − mean

- Δ: **-0.0206** [-0.0243, -0.0170]
- Verdict: **clearly_loses**

## Diagnostic — gen_on − gen_off (does lineup roll hurt?)

- Δ: **+0.0162** [+0.0116, +0.0209]
- Verdict: **clearly_wins**

## Diagnostic — gen_on − mean (current lineup stack vs mean)

- Δ: **-0.0043** [-0.0096, +0.0010]
- Verdict: **does_not_clearly_lose**

Artifacts: `data/processed/cascade_lineup_off/`

Reproduce:
```bash
PYTHONPATH=. python3 -m src.eval.cascade_lineup_off_tt1 --n-mc 20
```
