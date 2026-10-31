# Drivers of Opta−FD residual (2025-26)

Diagnostic on n=518 team-sides (GW8–38, ≥5 prior games). Dependent:
`resid = Opta λ − FD λ` (sd ≈ 0.40).

## Is form strong enough to lean on?

**No — not as a primary lever.**

| lens | form (L5 GD) |
| --- | ---: |
| univariate R² | **0.002** (raw ~flat) |
| ΔR² after venue + rank_gap + Opta margin | **+0.033** |
| partial r after that core | −0.24 |
| move per +1 SD form | −0.08 on resid (~20% of resid sd) |

Useful as a secondary explanation once place/venue are fixed; too weak and
too confounded to drive a default λ map on its own. L8 GD is slightly
stronger (ΔR² +0.041) than L5; form-vs-own-baseline is weak (ΔR² +0.004).

Early/late: ΔR² +0.044 (GW8–22) vs +0.025 (GW23–38) — present both halves,
larger early.

## Better drivers (univariate)

| feature | r with resid | R² |
| --- | ---: | ---: |
| Opta λ level | +0.54 | 0.29 |
| place rank_gap (opp−own) | +0.47 | 0.22 |
| Opta margin vs opp | +0.45 | 0.20 |
| home | −0.43 | 0.18 |
| is favourite (Opta) | +0.38 | 0.14 |
| L5 GD form | +0.04 | 0.00 |

## Nested R²

| model | R² |
| --- | ---: |
| venue only | 0.183 |
| venue + rank_gap | **0.405** |
| venue + Opta margin | 0.382 |
| venue + rank + opp_rank | 0.405 |
| + L5 GD form | 0.438 |
| + L5 pts | 0.425 |

**Dominant structure:** home/away misallocation plus strength mismatch
(favourites / better place vs opp). That is the same anatomy as the G
overprint isolate (Opta away-hot, favourite-hot), not a short-run form
story.

## Form definition (reference)

As-of mean over the last **5** completed matches before the GW: goal
difference / points / GF. No same-GW leakage.

## Verdict

Lean on **venue + strength mismatch** if shrinking Opta toward the book.
Treat form as a modest residual covariate, not the main signal.
Defaults unchanged.

Rows: `data/processed/cascade_long_fd_opta_form_resid_rows.csv`
Artifact: `/opt/cursor/artifacts/cascade_long_fd_opta_resid_drivers.json`
