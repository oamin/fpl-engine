Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# L4+Opta joint path — μ-band strand roll climb

Pre-registered: `protocol.l4_opta_joint_path_mu_band_roll`.

Roll each player's μ inside uncertainty bands (σ=`xp_se`, decayed with μ),
solve N independent joint paths, commit modal week-0 squad.

- Boards: `build_l3_centreline_horizon(H=12)` (wired μ, ρ=0.85→0 on μ+σ).
- Baseline `mean`: centreline μ (product).
- Treatment `mu_band_8`: N=8 strands, band ±2.0σ, modal week-0.
- Shared: joint path, captain, full board, `from_owned`, `ft_pe_nu=0`, no chips.
- Metric: paired GW realised XI(autosub)+cap/VC−hits. Wall 0s.

## Gates

- Engine as-of audit: **passed**.
- `score_xp_v2` as-of: **passed**.

## Results

### `mu_band_8` − `mean`

- Mean Δ pts/GW: **-2.500** [-4.575, -0.525]
- Covers 0: **False**
- Verdict: **park_clear_lose**

| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | base XI σ | treat XI σ | base buy σ | treat buy σ | modal/N | distinct | mean Δ/GW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 30 | 2087 | 1970 | -117 | 76 | 80 | 50 | 51 | 0.627 | 0.624 | 0.726 | 0.708 | 2.00/8 | 6.53 | -3.900 |
| 2023-24 | 30 | 1930 | 1855 | -75 | 72 | 108 | 49 | 58 | 0.556 | 0.562 | 0.657 | 0.687 | 2.50/8 | 6.33 | -2.500 |
| 2024-25 | 30 | 1904 | 1808 | -96 | 64 | 72 | 47 | 49 | 0.478 | 0.501 | 0.585 | 0.665 | 2.23/8 | 6.40 | -3.200 |
| 2025-26 | 30 | 1845 | 1833 | -12 | 28 | 48 | 38 | 43 | 0.454 | 0.465 | 0.467 | 0.530 | 3.27/8 | 5.30 | -0.400 |

## Notes

- **Park.** Clear lose vs product mean boards; all four seasons ≤0.
- Strand rolls reshuffle week-0 heavily (modal ~2–3/8; transfer agreement ≤10%) but
  do **not** systematically load high-σ names (XI σ ≈ flat; buy σ only mildly up in
  2023-24…2025-26). More hits under the band arm.
- Answer to the screen question: joint path under modal strand commit does **not**
  favour risky picks enough (or at all) to beat mean μ.
- Not cascade generative dice; not single-board μ±SE tilt.
- Path MILP time limit mixed across seasons (45–180s/strand); direction unchanged.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_mu_band_roll_gate
```

JSON: `data/processed/l4_opta_joint_path_mu_band_roll_gate.json`
