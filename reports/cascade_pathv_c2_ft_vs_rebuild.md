Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Cascade path-V C2 — FT path vs weekly free rebuild

Protocol: `cascade_pathv_c2_ft_vs_rebuild` (locked 2026-10-21).

## Claim

- **Treatment `ft_path`:** free rebuild open, then FT/hit continuity (`solve_week_transfers` on μ_post). Maps to `xi_plus_captain_ft` path rule.
- **Baseline `weekly_rebuild`:** free rebuild every GW (`solve_week_sigma`, no continuity). Maps to weekly `xi_plus_captain`.
- **Boards:** locked `score_xp_v2` μ_post (`mode=none`). Transfer-rule screen — generative boards parked after B1/B2 clearly_lose.
- **Metric:** paired-by-GW realised XI + captain − hits (`ft_path_minus_weekly_rebuild`).
- **Window:** GW8–38 · seasons 2022-23, 2023-24, 2024-25, 2025-26 · chips off.

## Gates

- As-of audit: **passed**.
- Paired intervals cover 2022-23…2025-26. 2026-27 holdout excluded.
- Protocol comparison `score_xp` − `score_exp_points` carried unchanged.

## Primary — `ft_path_minus_weekly_rebuild`

- Mean Δ/GW: **-9.145** [-11.783, -6.379] (cluster bootstrap B=1000, seed=0).
- Covers zero: **False** · clearly loses: **True** · clearly wins: **False**.
- **Verdict: `park_clearly_loses`**

### By season (net of hits)

| season | n GW | ft_path | weekly_rebuild | Δ | ft hit pts | mean hits/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 31 | 1968 | 2251 | -283 | 136 | 1.10 |
| 2023-24 | 31 | 1834 | 2090 | -256 | 140 | 1.13 |
| 2024-25 | 31 | 1960 | 2249 | -289 | 120 | 0.97 |
| 2025-26 | 31 | 1849 | 2155 | -306 | 100 | 0.81 |

**Pooled:** ft_path 7611 · rebuild 8745 · Δ **-1134** · ft hit pts **496**.

| season | mean Δ/GW |
| --- | ---: |
| 2022-23 | -9.129 |
| 2023-24 | -8.258 |
| 2024-25 | -9.323 |
| 2025-26 | -9.871 |

## Carried protocol comparison — `score_xp` − `score_exp_points`

- From `procedure_intervals.json`: **+0.0153** [+0.0130, +0.0175].
- Unchanged. Not re-estimated.

## Diagnostics (not win criteria)

- Wall-clock: **15.8s**.
- Solver time_limit: 20.0s · max_hits=15.

## Artefacts

- `data/processed/cascade_pathv_c2_intervals.json`
- `data/processed/cascade_pathv_c2/state.json`
- `reports/cascade_pathv_c2_ft_vs_rebuild.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_pathv_c2_ft_vs_rebuild
```
