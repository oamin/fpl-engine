Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path — scout-m share freeze vs recompute climb

Pre-registered: `protocol.l4_opta_joint_path_scout_m_share_policy`.

Authorised by likelihood `score_xp_v2_scout_m_share_policy` (promote).
Layer-1 isolation: `value_captain=false`, no chips, no σ.

- Boards: H=12, ρ=0.85→0, wired μ; scout m = rolling-3.
- Baseline `share_recompute`: sG/sA from scout m.
- Treatment `share_freeze`: sG/sA from expanding m; appear/CS/GC from scout m.
- Realised: autosub XI − hits. Wall ~102 min (4 season dumps).

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `share_freeze` − `share_recompute`

- Mean Δ pts/GW: **+0.883** [-1.209, +2.983]
- Covers 0: **True**
- Verdict: **inconclusive**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | base autosubs | treat autosubs | base blanks | treat blanks | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1637 | 1722 | +85 | 72 | 16 | 49 | 35 | 30 | 27 | 41 | 36 | +2.833 |
| 2023-24 | 30 | 1613 | 1671 | +58 | 72 | 28 | 49 | 38 | 31 | 28 | 42 | 42 | +1.933 |
| 2024-25 | 30 | 1573 | 1524 | -49 | 60 | 20 | 46 | 36 | 32 | 34 | 36 | 40 | -1.633 |
| 2025-26 | 30 | 1530 | 1542 | +12 | 20 | 8 | 36 | 33 | 31 | 30 | 48 | 43 | +0.400 |

## Notes

- Positive Δ = freeze shares beat recompute under scout rolling-3 m.
- Same scout m both arms; only the share law differs.
- No captain bank in realised metric (`value_captain=false`).

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_scout_m_share_policy_gate
```

JSON: `data/processed/l4_opta_joint_path_scout_m_share_policy_gate.json`
