Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path — component-μ climb (gated)

Pre-registered: `protocol.l4_opta_joint_path_component_climb`.

Planner isolation on locked L4 near / Opta far boards with `mu_post :=` one xP component. **Not** a μ promote.

- Boards: `build_l3_centreline_horizon(H=12)`; `mu_post :=` component.
- Baseline `l4_opta_h12_<C>`: `solve_horizon_transfers`.
- `joint_path_<C>`: joint path, full board, from_owned, undecayed.
- `joint_path_decay_<C>`: joint path + Opta far × ρ^(t−near+1) → 0 (ρ=0.85).
- Shared wildcard on full `score_xp_v2` @ GW8.
- No captaincy; no chips; XI−hits realised; max_hits=15; γ=1.0.
- Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

## `xp_assists`

### `joint_path` − `l4_opta_h12`

- Mean Δ pts/GW: **+0.117** [-1.426, +1.725]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1793 | 1695 | -98 | 4 | 0 | -3.267 |
| 2023-24 | 30 | 1570 | 1583 | +13 | 4 | 0 | +0.433 |
| 2024-25 | 30 | 1437 | 1509 | +72 | 8 | 0 | +2.400 |
| 2025-26 | 30 | 1468 | 1495 | +27 | 16 | 0 | +0.900 |

### `joint_path_decay` − `l4_opta_h12`

- Mean Δ pts/GW: **-0.150** [-1.917, +1.717]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1793 | 1761 | -32 | 4 | 0 | -1.067 |
| 2023-24 | 30 | 1570 | 1528 | -42 | 4 | 0 | -1.400 |
| 2024-25 | 30 | 1437 | 1445 | +8 | 8 | 0 | +0.267 |
| 2025-26 | 30 | 1468 | 1516 | +48 | 16 | 0 | +1.600 |

### `joint_path_decay` − `joint_path`

- Mean Δ pts/GW: **-0.267** [-1.642, +1.075]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1695 | 1761 | +66 | 0 | 0 | +2.200 |
| 2023-24 | 30 | 1583 | 1528 | -55 | 0 | 0 | -1.833 |
| 2024-25 | 30 | 1509 | 1445 | -64 | 0 | 0 | -2.133 |
| 2025-26 | 30 | 1495 | 1516 | +21 | 0 | 0 | +0.700 |

## `xp_cs`

### `joint_path` − `l4_opta_h12`

- Mean Δ pts/GW: **+1.542** [-0.783, +3.742]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1317 | 1224 | -93 | 12 | 0 | -3.100 |
| 2023-24 | 30 | 1240 | 1436 | +196 | 16 | 0 | +6.533 |
| 2024-25 | 30 | 1009 | 1214 | +205 | 20 | 0 | +6.833 |
| 2025-26 | 30 | 1372 | 1249 | -123 | 16 | 0 | -4.100 |

### `joint_path_decay` − `l4_opta_h12`

- Mean Δ pts/GW: **+1.917** [-0.309, +4.159]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1317 | 1216 | -101 | 12 | 0 | -3.367 |
| 2023-24 | 30 | 1240 | 1380 | +140 | 16 | 0 | +4.667 |
| 2024-25 | 30 | 1009 | 1205 | +196 | 20 | 0 | +6.533 |
| 2025-26 | 30 | 1372 | 1367 | -5 | 16 | 0 | -0.167 |

### `joint_path_decay` − `joint_path`

- Mean Δ pts/GW: **+0.375** [-1.567, +2.117]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1224 | 1216 | -8 | 0 | 0 | -0.267 |
| 2023-24 | 30 | 1436 | 1380 | -56 | 0 | 0 | -1.867 |
| 2024-25 | 30 | 1214 | 1205 | -9 | 0 | 0 | -0.300 |
| 2025-26 | 30 | 1249 | 1367 | +118 | 0 | 0 | +3.933 |

## `xp_gc`

### `joint_path` − `l4_opta_h12`

- Mean Δ pts/GW: **+0.767** [-0.851, +2.367]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 609 | 581 | -28 | 4 | 0 | -0.933 |
| 2023-24 | 30 | 517 | 657 | +140 | 8 | 0 | +4.667 |
| 2024-25 | 30 | 637 | 653 | +16 | 0 | 0 | +0.533 |
| 2025-26 | 30 | 624 | 588 | -36 | 0 | 0 | -1.200 |

### `joint_path_decay` − `l4_opta_h12`

- Mean Δ pts/GW: **+1.492** [-0.267, +3.295]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 609 | 600 | -9 | 4 | 0 | -0.300 |
| 2023-24 | 30 | 517 | 722 | +205 | 8 | 0 | +6.833 |
| 2024-25 | 30 | 637 | 629 | -8 | 0 | 0 | -0.267 |
| 2025-26 | 30 | 624 | 615 | -9 | 0 | 0 | -0.300 |

### `joint_path_decay` − `joint_path`

- Mean Δ pts/GW: **+0.725** [-0.742, +2.217]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 581 | 600 | +19 | 0 | 0 | +0.633 |
| 2023-24 | 30 | 657 | 722 | +65 | 0 | 0 | +2.167 |
| 2024-25 | 30 | 653 | 629 | -24 | 0 | 0 | -0.800 |
| 2025-26 | 30 | 588 | 615 | +27 | 0 | 0 | +0.900 |

## `xp_goals`

### `joint_path` − `l4_opta_h12`

- Mean Δ pts/GW: **-1.417** [-3.417, +0.635]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1641 | 1546 | -95 | 12 | 0 | -3.167 |
| 2023-24 | 30 | 1527 | 1478 | -49 | 12 | 0 | -1.633 |
| 2024-25 | 30 | 1566 | 1528 | -38 | 12 | 0 | -1.267 |
| 2025-26 | 30 | 1390 | 1402 | +12 | 20 | 4 | +0.400 |

### `joint_path_decay` − `l4_opta_h12`

- Mean Δ pts/GW: **-0.833** [-2.925, +1.233]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1641 | 1603 | -38 | 12 | 0 | -1.267 |
| 2023-24 | 30 | 1527 | 1480 | -47 | 12 | 0 | -1.567 |
| 2024-25 | 30 | 1566 | 1559 | -7 | 12 | 0 | -0.233 |
| 2025-26 | 30 | 1390 | 1382 | -8 | 20 | 0 | -0.267 |

### `joint_path_decay` − `joint_path`

- Mean Δ pts/GW: **+0.583** [-0.909, +2.043]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1546 | 1603 | +57 | 0 | 0 | +1.900 |
| 2023-24 | 30 | 1478 | 1480 | +2 | 0 | 0 | +0.067 |
| 2024-25 | 30 | 1528 | 1559 | +31 | 0 | 0 | +1.033 |
| 2025-26 | 30 | 1402 | 1382 | -20 | 4 | 0 | -0.667 |

## Notes

- Complements `l4_component_horizon_climb` (H=12 vs H=2 on L4 oneshot) and full-μ `l4_opta_joint_path_climb`.
- Season totals are descriptive; the gate is the paired GW interval.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_component_climb_gate
```

JSON: `data/processed/l4_opta_joint_path_component_climb_gate.json`
