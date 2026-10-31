Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Joint path μ-wire climb sanity — gated

Pre-registered: `protocol.l4_opta_joint_path_mu_wire_climb`.

After likelihood promotes for rate baseline + full-club shares, sanity-check locked `joint_path_opta_decay` on wired boards.

- `joint_path_decay_legacy`: prior `joint_path_decay` dumps (pre-wire μ).
- `l4_opta_h12_wired`: L4 oneshot on wired centreline μ.
- `joint_path_decay_wired`: joint path + Opta far ρ=0.85→0 on wired μ.
- Metric: paired GW XI−hits. Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `joint_path_decay_wired` − `joint_path_decay_legacy` (μ wire)

- Mean Δ pts/GW: **-0.250** [-2.167, +1.725]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1700 | 1695 | -5 | -0.167 |
| 2023-24 | 30 | 1726 | 1739 | +13 | +0.433 |
| 2024-25 | 30 | 1639 | 1616 | -23 | -0.767 |
| 2025-26 | 30 | 1633 | 1618 | -15 | -0.500 |

### `joint_path_decay_wired` − `l4_opta_h12_wired` (planner)

- Mean Δ pts/GW: **+1.125** [-0.667, +2.951]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1696 | 1695 | -1 | -0.033 |
| 2023-24 | 30 | 1650 | 1739 | +89 | +2.967 |
| 2024-25 | 30 | 1529 | 1616 | +87 | +2.900 |
| 2025-26 | 30 | 1658 | 1618 | -40 | -1.333 |

## Notes

- Positive μ-wire Δ = wired boards beat pre-wire dumps under the same joint-path decay strategiser.
- Positive planner Δ = joint path still beats L4 oneshot on wired μ.
- Park the wire if μ-wire interval hi < 0 despite likelihood promote.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_mu_wire_climb_gate
```

JSON: `data/processed/l4_opta_joint_path_mu_wire_climb_gate.json`
