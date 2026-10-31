Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path climb — gated

Pre-registered: `protocol.l4_opta_joint_path_climb`.

Planner isolation on locked L4 near / Opta far centreline μ. **Not** a μ promote.

- Boards: `build_l3_centreline_horizon(H=12)` (priced t:t+1 + Opta far).
- Baseline `l4_opta_h12`: `solve_horizon_transfers` (t=0 transfers only).
- `joint_path`: joint `pick_path_transfer_plan`, full board, `from_owned`, undecayed.
- `joint_path_decay`: same planner; Opta far μ × ρ^(t−near+1) → 0 (ρ=0.85, near=2).
- Shared wildcard open on `score_xp_v2` @ GW8; seed held 15.
- No captaincy (objective + realised XI only); no chips; hits uncapped (max_hits=15, −4/hit); γ=1.0.
- Metric: paired GW realised XI−hits (transfer weeks). Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `joint_path` − `l4_opta_h12` (planner)

- Mean Δ pts/GW: **+1.925** [+0.083, +3.692]
- Covers 0: **False**
- Verdict: **promote**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1660 | 1818 | +158 | 60 | 88 | 46 | 53 | +5.267 |
| 2023-24 | 30 | 1674 | 1741 | +67 | 64 | 112 | 47 | 59 | +2.233 |
| 2024-25 | 30 | 1608 | 1576 | -32 | 76 | 84 | 50 | 52 | -1.067 |
| 2025-26 | 30 | 1542 | 1580 | +38 | 84 | 76 | 52 | 50 | +1.267 |

### `joint_path_decay` − `l4_opta_h12`

- Mean Δ pts/GW: **+1.783** [-0.150, +3.884]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1660 | 1700 | +40 | 60 | 88 | 46 | 53 | +1.333 |
| 2023-24 | 30 | 1674 | 1726 | +52 | 64 | 100 | 47 | 56 | +1.733 |
| 2024-25 | 30 | 1608 | 1639 | +31 | 76 | 88 | 50 | 53 | +1.033 |
| 2025-26 | 30 | 1542 | 1633 | +91 | 84 | 64 | 52 | 47 | +3.033 |

### `joint_path_decay` − `joint_path` (decay)

- Mean Δ pts/GW: **-0.142** [-1.609, +1.359]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1818 | 1700 | -118 | 88 | 88 | 53 | 53 | -3.933 |
| 2023-24 | 30 | 1741 | 1726 | -15 | 112 | 100 | 59 | 56 | -0.500 |
| 2024-25 | 30 | 1576 | 1639 | +63 | 84 | 88 | 52 | 53 | +2.100 |
| 2025-26 | 30 | 1580 | 1633 | +53 | 76 | 64 | 50 | 47 | +1.767 |

## Notes

- Positive planner Δ = joint every-GW path beats L4 oneshot on the same undecayed L4+Opta boards.
- Positive decay Δ vs `joint_path` = damping Opta far xp toward 0 helps the joint planner.
- Season totals are descriptive; the gate is the paired GW interval.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_climb_gate
```

JSON: `data/processed/l4_opta_joint_path_climb_gate.json`
