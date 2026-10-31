Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path — captain double climb

Pre-registered: `protocol.l4_opta_joint_path_captain_climb`.

Reintroduce captain in the path objective on locked `joint_path_opta_decay`.
Realised for **both** arms: FPL autosubs + captain/VC − hits.

- Boards: `build_l3_centreline_horizon(H=12)` (wired μ, ρ=0.85→0).
- Baseline `decay_xi_ft`: `value_captain=false` (product).
- Treatment `decay_xi_plus_captain`: `value_captain=true` (Σ XI+cap − hits).
- Shared: joint path, full board, `from_owned`, `ft_pe_nu=0`, no chips.
- Metric: paired GW realised XI(autosub)+cap/VC−hits. Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `decay_xi_plus_captain` − `decay_xi_ft`

- Mean Δ pts/GW: **+3.483** [+1.866, +5.267]
- Covers 0: **False**
- Verdict: **promote**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | base cap | treat cap | base autosubs | treat autosubs | base VC wks | treat VC wks | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 1868 | 2087 | +219 | 72 | 76 | 49 | 50 | 173 | 249 | 0 | 0 | 0 | 0 | +7.300 |
| 2023-24 | 30 | 1922 | 1936 | +14 | 52 | 72 | 44 | 49 | 183 | 194 | 0 | 0 | 0 | 0 | +0.467 |
| 2024-25 | 30 | 1776 | 1904 | +128 | 28 | 64 | 38 | 47 | 160 | 289 | 0 | 0 | 0 | 0 | +4.267 |
| 2025-26 | 30 | 1788 | 1845 | +57 | 36 | 28 | 40 | 38 | 170 | 208 | 0 | 0 | 0 | 0 | +1.900 |

## Notes

- Positive Δ = valuing captain in V beats product xi_ft on autosub+VC realised.
- Both arms use the same realised metric (objective-only contrast).
- VC doubles only when the intended captain records 0 minutes.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_captain_climb_gate
```

JSON: `data/processed/l4_opta_joint_path_captain_climb_gate.json`
