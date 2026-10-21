# Path FT potential energy vs legacy objective (joint plan)

Protocol: `cascade_ft_pe_path` · season **2025-26** · decision GW6 → 8–19 · N=12 · seed=7

**Diagnostic retest under `cascade_joint_path_plan`.** Same path setup as `cascade_mean_path_gw_resid` (full-side shares, semi-oracle buy, generative attack, autosub actuals). Planner is one joint multi-GW FT schedule per strand (not the retired myopic snapshot FT MILP).

| arm | objective |
| --- | --- |
| control | `V = Σ (XI+cap) − hit_pts` (`ft_pe_nu=0`, γ=1) |
| candidate | `V + U(f_H)`, `U(b)=ν·b`, **ν=4** (=HIT_COST): each stored FT = +4, five = +20 (terminal bank) |

Bank dynamics: hold below cap → next deadline bank +1 → PE +4; at five FTs the weekly grant adds +0 PE unless a transfer is used first. Retired: concave φ and multi-week W scaling.

Wall: 114.1s

## Transfer behaviour (mean over strands; excl. opening free rebuild)

| arm | transfers | free used | paid | hit pts | mean terminal ft |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 26.58 | 11.00 | 15.58 | 62.3 | 1.00 |
| candidate | 26.50 | 10.92 | 15.58 | 62.3 | 1.08 |
| Δ (cand−ctrl) | -0.08 |  | +0.00 | +0.0 | +0.08 |

### Per GW (candidate joint schedule)

| GW | transfers | free | paid | hit pts | mean ft_after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 1.33 | 0.83 | 0.50 | 2.0 | 1.17 |
| 10 | 2.67 | 1.17 | 1.50 | 6.0 | 1.00 |
| 11 | 2.83 | 1.00 | 1.83 | 7.3 | 1.00 |
| 12 | 2.67 | 1.00 | 1.67 | 6.7 | 1.00 |
| 13 | 2.67 | 1.00 | 1.67 | 6.7 | 1.00 |
| 14 | 2.42 | 1.00 | 1.42 | 5.7 | 1.00 |
| 15 | 2.33 | 0.92 | 1.42 | 5.7 | 1.08 |
| 16 | 2.33 | 1.08 | 1.25 | 5.0 | 1.00 |
| 17 | 2.92 | 1.00 | 1.92 | 7.7 | 1.00 |
| 18 | 2.42 | 0.92 | 1.50 | 6.0 | 1.08 |
| 19 | 1.92 | 1.00 | 0.92 | 3.7 | 1.08 |

## Path residual vs actual (oracle minutes + autosub)

| arm | sum actual | resid bias | MAE |
| --- | ---: | ---: | ---: |
| control | 629.7 | -2.94 | 8.33 |
| candidate | 630.2 | -2.91 | 8.24 |
| Δ | +0.6 | +0.02 |  |

## Decision

**`keep_watching_pe_small`** — under the joint plan, terminal PE (ν=4) barely moves transfer behaviour; path actual Δ=+0.6.

Prior myopic-snapshot PE screen is superseded by this joint-plan retest.


## Artefacts

- `data/processed/cascade_ft_pe_path/state.json`
- `data/predictions/2025-26/cascade_ft_pe_path`
- `reports/cascade_ft_pe_path.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_ft_pe_path --season 2025-26 --n 12
```
