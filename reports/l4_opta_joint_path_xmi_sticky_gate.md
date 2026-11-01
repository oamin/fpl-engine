Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path — sticky xmi minutes climb

Pre-registered: `protocol.l4_opta_joint_path_xmi_sticky`.

Layer-1 isolation: `value_captain=false`, no chips, no σ. Scout-like
as-of minutes portfolio vs product expanding mean.

- Boards: H=12, ρ=0.85→0, wired μ.
- Baseline `expanding`: expanding as-of mean m.
- Treatment `xmi_sticky`: rolling-3; last GW blank → m=0 on all boards.
- Realised: autosub XI − hits. Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `xmi_sticky` − `expanding`

- Mean Δ pts/GW: **-2.558** [-4.892, -0.200]
- Covers 0: **False**
- Verdict: **park_clear_lose**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | base autosubs | treat autosubs | base blanks | treat blanks | sticky0 mean | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1695 | 1607 | -88 | 72 | 84 | 49 | 52 | 0 | 20 | 2 | 29 | 409.5 | -2.933 |
| 2023-24 | 30 | 1739 | 1599 | -140 | 52 | 72 | 44 | 49 | 0 | 27 | 1 | 35 | 497.6 | -4.667 |
| 2024-25 | 30 | 1582 | 1559 | -23 | 24 | 56 | 37 | 45 | 0 | 30 | 0 | 34 | 427.2 | -0.767 |
| 2025-26 | 30 | 1618 | 1562 | -56 | 36 | 60 | 40 | 46 | 0 | 30 | 0 | 42 | 492.2 | -1.867 |

## Notes

- **Park.** Clear lose; all four seasons ≤0.
- One-blank sticky is too harsh: ~410–500 pool zeros/GW, more hits, blanks 0–2→29–42,
  autosubs appear but net still down. Expanding mean keeps rebounders.
- Oracle@t (+2.07) ceiling remains — this as-of proxy does **not** close it.
- Live scout should carry tagged “out N weeks”, not last-blank→0.
- No captain bank (`value_captain=false`).

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_xmi_sticky_gate
```

JSON: `data/processed/l4_opta_joint_path_xmi_sticky_gate.json`
