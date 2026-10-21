# Path FT potential energy vs legacy objective (joint plan)

Protocol: `cascade_ft_pe_path` · season **2025-26** · decision GW6 → 8–19 · N=12 · seed=7

**Diagnostic retest under `cascade_joint_path_plan`.** Same path setup as `cascade_mean_path_gw_resid` (full-side shares, semi-oracle buy, generative attack, autosub actuals). Planner is one joint multi-GW FT schedule per strand (not the retired myopic snapshot FT MILP).

| arm | objective |
| --- | --- |
| control | `V = Σ (XI+cap) − hit_pts` (`ft_pe_nu=0`, **γ=1**) |
| candidate | `V + U(f_H)`, `U(b)=ν·φ(b)·W`, **ν=2** (=HIT_COST/2), W=H under γ=1 (terminal bank only) |

φ(b) = Σᵢ₌₁..b (1 − (i−1)/(2·MAX_FT)): φ(1)=1.00, φ(2)=1.90, φ(3)=2.70, φ(4)=3.40, φ(5)=4.00

W≈12.00 for H=12, γ=1.0. Under the joint plan PE is terminal leftover-bank value; mid-horizon banking is already priced via future XI scores.

Wall: 100.8s

## Transfer behaviour (mean over strands; excl. opening free rebuild)

| arm | transfers | free used | paid | hit pts | mean terminal ft |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 27.00 | 11.00 | 16.00 | 64.0 | 1.00 |
| candidate | 27.00 | 11.00 | 16.00 | 64.0 | 1.00 |
| Δ (cand−ctrl) | +0.00 |  | +0.00 | +0.0 | +0.00 |

### Per GW (candidate joint schedule)

| GW | transfers | free | paid | hit pts | mean ft_after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 1.58 | 1.00 | 0.58 | 2.3 | 1.00 |
| 10 | 2.33 | 1.00 | 1.33 | 5.3 | 1.00 |
| 11 | 3.08 | 1.00 | 2.08 | 8.3 | 1.00 |
| 12 | 2.75 | 1.00 | 1.75 | 7.0 | 1.00 |
| 13 | 2.50 | 1.00 | 1.50 | 6.0 | 1.00 |
| 14 | 2.50 | 1.00 | 1.50 | 6.0 | 1.00 |
| 15 | 2.50 | 1.00 | 1.50 | 6.0 | 1.00 |
| 16 | 2.33 | 1.00 | 1.33 | 5.3 | 1.00 |
| 17 | 2.92 | 1.00 | 1.92 | 7.7 | 1.00 |
| 18 | 2.58 | 1.00 | 1.58 | 6.3 | 1.00 |
| 19 | 1.92 | 1.00 | 0.92 | 3.7 | 1.00 |

## Path residual vs actual (oracle minutes + autosub)

| arm | sum actual | resid bias | MAE |
| --- | ---: | ---: | ---: |
| control | 630.7 | -3.11 | 8.64 |
| candidate | 630.8 | -3.08 | 8.62 |
| Δ | +0.1 | +0.02 |  |

## Decision

**`keep_watching_pe_small`** — under the joint plan, terminal PE (ν=2) barely moves transfer behaviour; path actual Δ=+0.1.

Prior myopic-snapshot PE screen is superseded by this joint-plan retest.


## Artefacts

- `data/processed/cascade_ft_pe_path/state.json`
- `data/predictions/2025-26/cascade_ft_pe_path`
- `reports/cascade_ft_pe_path.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_ft_pe_path --season 2025-26 --n 12
```
