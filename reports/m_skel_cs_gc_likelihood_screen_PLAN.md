# Plan — same-skeleton discrete CS/GC likelihood (pre-registered)

**Protocol:** `m_skel_cs_gc_likelihood_screen`  
**Pre-registered:** 2026-10-30  
**Status:** done_promote

## Motivation

The prior screen (`m_discrete_event_likelihood_screen`) compared a five-term
factored score to **one** Gaussian on totals and clearly lost for scale
reasons. A post-hoc diagnostic kept the same five-term skeleton and only
swapped CS/GC laws; that comparison was never the promote gate, and the
pooled headline was inflated by MID/FWD rows (discrete CS/GC contribute 0;
Gaussian still pays \(N(0\mid0,\sigma)\)).

This screen is the clean re-registration of that claim.

## Claim

On **GKP/DEF** player-GWs, a discrete CS/GC predictive law from Dixon–Coles
\(M\) beats Gaussian CS/GC around the locked centreline means, under an
identical five-term skeleton (G/A/rest unchanged). Mean layer untouched.

## Arms

| Arm | Predictive model |
| --- | --- |
| **Baseline** `l3_skel_gauss_cs_gc` | \(\log p=\log p_G+\log p_A+\log p_{CS}^{\mathrm{N}}+\log p_{GC}^{\mathrm{N}}+\log p_{\mathrm{rest}}\) |
| **Treatment** `l3_skel_disc_cs_gc` | \(\log p=\log p_G+\log p_A+\log p_{CS}^{\mathrm{Bern}}+\log p_{GC}^{\mathrm{cat}}+\log p_{\mathrm{rest}}\) |

Identical on both arms (locked FD centreline `score_xp_v2` μ, σ=3):

- \(p_G\): Gaussian(`xp_goals`)
- \(p_A\): Gaussian(`xp_assists`)
- \(p_{\mathrm{rest}}\): Gaussian(`xp − xp_G − xp_A − xp_cs − xp_gc`) vs actual rest points

CS/GC only:

| Factor | Baseline | Treatment |
| --- | --- | --- |
| CS | Gaussian(`P_CS` \| `xp_cs`, σ) | Bernoulli(`P(CS)` from \(M\)) on \(\{P_{CS}>0\}\) |
| GC | Gaussian(`P_GC` \| `xp_gc`, σ) | categorical \(P(\lfloor GA/2\rfloor=k)\) from \(M\) |

No fitted β. No mode/median μ. No Opta. No MID/FWD in the pool.

## Pool & seasons

- Positions: **GKP, DEF** only
- GW **8–38**, four **closed** seasons (2022-23…2025-26)
- Eligible: **minutes > 60**; **≥ 3** prior apps as-of GW; FD AH+OU priced
- **2026/27 holdout frozen** — not in the comparison

## Metric & gates

1. As-of audit must pass.
2. Primary: mean log predictive density per player-GW; paired **treatment − baseline** by GW; `cluster_interval` across four closed seasons.
3. **Promote if** paired interval **lo > 0**. Else park; mean layer unchanged either way.
4. `write_gated_report` only.

## Explicitly out of scope

- Changing `score_team_side` / α / shares / venue shrinks
- Replacing μ with mode/median boards
- Full-squad (MID/FWD) same-skeleton (prior diagnostic artifact)
- Factored-vs-Gaussian-total (already parked)
- Captain / chips / OF search

## Deliverables

| Artifact | Path |
| --- | --- |
| Runner | `src/eval/m_skel_cs_gc_likelihood_screen.py` |
| Gated report | `reports/m_skel_cs_gc_likelihood_screen.md` |
| State | `REPORT_STATE.json` |

## Reproduce (after implement)

```bash
PYTHONPATH=. python3 -m src.eval.m_skel_cs_gc_likelihood_screen
```

## Decision tree

```
run gated screen
├─ clearly_wins (lo>0) → keep mean board; adopt discrete CS/GC in predictive
│                        likelihood / eval; optional DEF pick sanity later
├─ inconclusive → park; leave Gaussian CS/GC in factored eval if any
└─ clearly_loses → park discreteness claim for CS/GC under this skeleton
```
