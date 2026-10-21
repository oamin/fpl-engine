# Path FT potential energy vs legacy objective

Protocol: `cascade_ft_pe_path` · season **2025-26** · decision GW6 → 8–19 · N=12 · seed=7

**Diagnostic.** Same path setup as `cascade_mean_path_gw_resid` (full-side shares, semi-oracle buy, generative attack, autosub actuals).

| arm | objective |
| --- | --- |
| control | `V = S − hit_pts` (`ft_pe_nu=0`) |
| candidate | `V = S − hit_pts + U(ft_after)`, `U(b)=ν·φ(b)·W`, **ν=2** (=HIT_COST/2), W=Σγ^h on remaining horizon |

φ(b) = Σᵢ₌₁..b (1 − (i−1)/(2·MAX_FT)): φ(1)=1.00, φ(2)=1.90, φ(3)=2.70, φ(4)=3.40, φ(5)=4.00

First smoke without W left transfers unchanged (S is multi-week). W is the unit alignment, not a fit.

Wall: 179.3s

## Transfer behaviour (mean over strands; excl. GW8 free rebuild)

| arm | transfers | free used | paid | hit pts |
| --- | ---: | ---: | ---: | ---: |
| control | 19.08 | 10.00 | 9.08 | 36.3 |
| candidate | 18.75 | 10.00 | 8.75 | 35.0 |
| Δ (cand−ctrl) | -0.33 |  | -0.33 | -1.3 |

### Per snapshot (candidate)

| GW | transfers | free | paid | hit pts | mean ft_after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 3.83 | 2.00 | 1.83 | 7.3 | 1.00 |
| 12 | 3.75 | 1.92 | 1.83 | 7.3 | 1.08 |
| 14 | 3.67 | 2.00 | 1.67 | 6.7 | 1.08 |
| 16 | 3.42 | 1.92 | 1.50 | 6.0 | 1.17 |
| 18 | 4.08 | 2.17 | 1.92 | 7.7 | 1.00 |

## Path residual vs actual (oracle minutes + autosub)

| arm | sum actual | resid bias | MAE |
| --- | ---: | ---: | ---: |
| control | 646.4 | +1.06 | 11.10 |
| candidate | 648.1 | +0.68 | 11.36 |
| Δ | +1.7 | -0.38 |  |

## Decision

**`keep_watching_pe_small`** — diagnostic only, not a four-season promote.

With `U=ν·φ·W` (ν=2), paid transfers fall only **−0.33/strand** and hit pts **−1.3**, while path actual is **+1.7** (not worse). The path still burns ~2 FT + ~2 hits most snapshots — generative multi-week `S` gaps still dominate HIT_COST-scale PE. Next natural step if pursued: strand **continuation** `U(b)` (DP/MC), not a larger ν fit.

## Artefacts

- `data/processed/cascade_ft_pe_path/state.json`
- `data/predictions/2025-26/cascade_ft_pe_path`
- `reports/cascade_ft_pe_path.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_ft_pe_path --season 2025-26 --n 12
```
