Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path — Opta-far decay ρ / target tune

Pre-registered: `protocol.l4_opta_joint_path_decay_tune`.

Strategiser knobs on locked `joint_path_opta_decay` boards/planner. **Not** a μ promote.

- Boards: `build_l3_centreline_horizon(H=12)`.
- Baseline `decay_r085`: ρ=0.85 → 0 (locked product).
- `decay_r070`: ρ=0.70 → 0.
- `decay_r095`: ρ=0.95 → 0.
- `decay_r085_posmean`: ρ=0.85 → positional mean of each xp column.
- Shared: joint path, full board, `from_owned`, no captaincy, no chips.
- Metric: paired GW realised XI−hits. Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `decay_r070` − `decay_r085`

- Mean Δ pts/GW: **-0.617** [-1.917, +0.717]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1700 | 1661 | -39 | 88 | 96 | 53 | 55 | -1.300 |
| 2023-24 | 30 | 1726 | 1684 | -42 | 100 | 108 | 56 | 58 | -1.400 |
| 2024-25 | 30 | 1639 | 1655 | +16 | 88 | 80 | 53 | 51 | +0.533 |
| 2025-26 | 30 | 1633 | 1624 | -9 | 64 | 52 | 47 | 44 | -0.300 |

### `decay_r095` − `decay_r085`

- Mean Δ pts/GW: **-0.333** [-1.675, +1.117]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1700 | 1710 | +10 | 88 | 92 | 53 | 54 | +0.333 |
| 2023-24 | 30 | 1726 | 1748 | +22 | 100 | 116 | 56 | 60 | +0.733 |
| 2024-25 | 30 | 1639 | 1619 | -20 | 88 | 76 | 53 | 50 | -0.667 |
| 2025-26 | 30 | 1633 | 1581 | -52 | 64 | 60 | 47 | 46 | -1.733 |

### `decay_r085_posmean` − `decay_r085`

- Mean Δ pts/GW: **-0.400** [-1.833, +1.109]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1700 | 1763 | +63 | 88 | 64 | 53 | 47 | +2.100 |
| 2023-24 | 30 | 1726 | 1701 | -25 | 100 | 120 | 56 | 61 | -0.833 |
| 2024-25 | 30 | 1639 | 1631 | -8 | 88 | 76 | 53 | 50 | -0.267 |
| 2025-26 | 30 | 1633 | 1555 | -78 | 64 | 64 | 47 | 47 | -2.600 |

## Notes

- Positive Δ = treatment beats locked ρ=0.85→0 on realised XI−hits.
- Season totals are descriptive; the gate is the paired GW interval.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_decay_tune_gate
```

JSON: `data/processed/l4_opta_joint_path_decay_tune_gate.json`
