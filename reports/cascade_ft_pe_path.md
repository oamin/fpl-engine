# Path FT potential energy vs legacy objective

Protocol: `cascade_ft_pe_path` · season **2025-26** · decision GW6 → 8–19 · N=12 · seed=7

**Diagnostic.** Same path setup as `cascade_mean_path_gw_resid` (full-side shares, semi-oracle buy, generative attack, autosub actuals).

| arm | objective |
| --- | --- |
| control | `V = S − hit_pts` (`ft_pe_nu=0`) |
| candidate | `V = S − hit_pts + U(ft_after)`, `U(b)=ν·φ(b)`, **ν=2** (=HIT_COST/2) |

φ(b) = Σᵢ₌₁..b (1 − (i−1)/(2·MAX_FT)): φ(1)=1.00, φ(2)=1.90, φ(3)=2.70, φ(4)=3.40, φ(5)=4.00

Wall: 179.3s

## Transfer behaviour (mean over strands; excl. GW8 free rebuild)

| arm | transfers | free used | paid | hit pts |
| --- | ---: | ---: | ---: | ---: |
| control | 19.08 | 10.00 | 9.08 | 36.3 |
| candidate | 19.08 | 10.00 | 9.08 | 36.3 |
| Δ (cand−ctrl) | +0.00 |  | +0.00 | +0.0 |

### Per snapshot (candidate)

| GW | transfers | free | paid | hit pts | mean ft_after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 3.83 | 2.00 | 1.83 | 7.3 | 1.00 |
| 12 | 3.92 | 2.00 | 1.92 | 7.7 | 1.00 |
| 14 | 3.83 | 2.00 | 1.83 | 7.3 | 1.00 |
| 16 | 3.58 | 2.00 | 1.58 | 6.3 | 1.00 |
| 18 | 3.92 | 2.00 | 1.92 | 7.7 | 1.00 |

## Path residual vs actual (oracle minutes + autosub)

| arm | sum actual | resid bias | MAE |
| --- | ---: | ---: | ---: |
| control | 646.4 | +1.06 | 11.10 |
| candidate | 646.4 | +1.06 | 11.10 |
| Δ | +0.0 | +0.00 |  |

## Decision

Diagnostic only — not a four-season promote. PE is working if paid transfers / hit pts fall without a clear drop in path actual.

## Artefacts

- `data/processed/cascade_ft_pe_path/state.json`
- `data/predictions/2025-26/cascade_ft_pe_path`
- `reports/cascade_ft_pe_path.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_ft_pe_path --season 2025-26 --n 12
```
