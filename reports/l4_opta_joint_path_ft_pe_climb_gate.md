Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path — FT potential energy climb

Pre-registered: `protocol.l4_opta_joint_path_ft_pe_climb`.

Terminal FT PE on locked `joint_path_opta_decay` (wired μ, ρ=0.85→0).
PE is objective-only; realised remains XI−hits.

- Boards: `build_l3_centreline_horizon(H=12)` (wired rate baseline + full-club shares).
- Baseline `decay_ft_pe0`: `ft_pe_nu=0` (product).
- Treatment `decay_ft_pe4`: `ft_pe_nu=HIT_COST` (+4 per terminal banked FT).
- Shared: joint path, full board, `from_owned`, no captaincy, no chips, ρ=0.85→0.
- Metric: paired GW realised XI−hits. Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `decay_ft_pe4` − `decay_ft_pe0`

- Mean Δ pts/GW: **+0.108** [-1.259, +1.501]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | base mean ft_after | treat mean ft_after | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1695 | 1806 | +111 | 72 | 48 | 49 | 43 | 0.00 | 1.07 | +3.700 |
| 2023-24 | 30 | 1739 | 1724 | -15 | 52 | 64 | 44 | 46 | 0.00 | 1.23 | -0.500 |
| 2024-25 | 30 | 1616 | 1512 | -104 | 28 | 36 | 38 | 36 | 0.00 | 1.40 | -3.467 |
| 2025-26 | 30 | 1618 | 1639 | +21 | 36 | 28 | 40 | 38 | 0.00 | 1.27 | +0.700 |

## Notes

- Positive Δ = ν=HIT_COST beats product ν=0 on realised XI−hits.
- Season totals are descriptive; the gate is the paired GW interval.
- Reused wired `joint_path_decay` dumps as baseline when `--reuse-baseline` set.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_ft_pe_climb_gate
```

JSON: `data/processed/l4_opta_joint_path_ft_pe_climb_gate.json`
