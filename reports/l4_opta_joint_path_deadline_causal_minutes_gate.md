Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path — deadline-causal minutes climb

Pre-registered: `protocol.l4_opta_joint_path_deadline_causal_minutes`.

Oracle minutes at decision GW **t** vs product as-of mean, on locked
`joint_path_opta_decay` + captain. Far horizon stays as-of (GW `< t`).

- Boards: `build_l3_centreline_horizon(H=12)`; ρ=0.85→0; `value_captain=true`.
- Baseline `asof_all`: week-0 μ rebuilt with **as-of** mean minutes.
- Treatment `oracle_t`: week-0 μ rebuilt with **oracle** (actual) minutes at t.
- Far boards (both): as-of mean minutes.
- Realised (both): autosub + captain/VC − hits.
- Metric: paired GW realised. Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `oracle_t` − `asof_all`

- Mean Δ pts/GW: **+2.067** [+0.325, +3.875]
- Covers 0: **False**
- Verdict: **promote**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | base cap | treat cap | base autosubs | treat autosubs | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1937 | 1980 | +43 | 12 | 44 | 34 | 42 | 233 | 237 | 28 | 2 | +1.433 |
| 2023-24 | 30 | 1840 | 1851 | +11 | 16 | 64 | 35 | 47 | 199 | 205 | 36 | 2 | +0.367 |
| 2024-25 | 30 | 1712 | 1927 | +215 | 16 | 36 | 35 | 40 | 271 | 294 | 32 | 1 | +7.167 |
| 2025-26 | 30 | 1770 | 1749 | -21 | 0 | 36 | 31 | 40 | 186 | 172 | 41 | 3 | -0.700 |

## Notes

## Product implication

**Promote = week-t minutes information value**, not a product wire.

Actual minutes at decision GW are not known pre-deadline. Wiring them into
hist μ would leak outcomes. Keep product boards on **as-of** minutes.

Oracle collapses autosubs (137→8 across seasons) and raises hit spend — the
planner trusts week-t involvement and moves more. The gap to close live is
better week-t xmi / lineup news toward this ceiling.


- Positive Δ = oracle@t week-0 beats as-of week-0 on realised autosub+VC−hits.
- Prior diagnostic: `cascade_deadline_causal_minutes` (N=12, not a promote gate).

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_deadline_causal_minutes_gate
```

JSON: `data/processed/l4_opta_joint_path_deadline_causal_minutes_gate.json`
