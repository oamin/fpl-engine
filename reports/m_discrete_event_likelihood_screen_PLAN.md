# Plan — discrete M-event predictive likelihood (pre-registered)

**Protocol:** `m_discrete_event_likelihood_screen`  
**Pre-registered:** 2026-10-30  
**Status:** registered (no totals read yet)

## Motivation

Integrated Dixon–Coles \(M\) (Σ-over-\(M\) → `score_team_side`) is the right
**mean** board: stable, season-calibrated, and already uses
\(\alpha_G/\alpha_A\) for OGs / no-assist.

Mode / median scorelines are a bad discreteness proxy (Arsenal favourites
freeze at GA∈{0,1} → GC component stuck at 0). Discreteness belongs in the
**predictive distribution**, not in a different point μ.

## Claim

A factored predictive likelihood that uses \(M\)’s discrete CS / GC structure
beats the current Gaussian-on-total around locked centreline μ, on player-GW
log score, without changing `score_xp_v2` means.

## Arms

| Arm | Predictive model |
| --- | --- |
| **Baseline** `l3_gaussian_total` | \(\log p(y)=\texttt{gaussian_log_score}(y,\mu,\sigma)\) with \(\mu=\) FD centreline `score_xp_v2` `xp`, \(\sigma=3\) |
| **Treatment** `l3_factored_m_events` | \(\log p(y)=\log p_G+\log p_A+\log p_{CS}+\log p_{GC}+\log p_{\mathrm{rest}}\) using the **same** locked μ components |

### Treatment factor detail (locked)

All from the same FD λ → \(M\) → `score_team_side` board (as-of):

| Factor | Law | Target |
| --- | --- | --- |
| \(p_G\) | Gaussian(`xp_goals`, σ) | actual `P(G)` |
| \(p_A\) | Gaussian(`xp_assists`, σ) | actual `P(A)=3×assists` |
| \(p_{CS}\) | Bernoulli(`xcs`\(=P(\mathrm{CS})\)) on event \(\{P_{CS}>0\}\); density on points = that Bernoulli (MID/FWD use `CS_POINTS`) | actual CS award > 0 |
| \(p_{GC}\) | For GKP/DEF: categorical \(P(\lfloor GA/2\rfloor=k)\) from \(M\) marginal, score \(-k\); else point mass at 0 | actual `P(GC)` |
| \(p_{\mathrm{rest}}\) | Gaussian(`xp − xp_G − xp_A − xp_cs − xp_gc`, σ) | `total_points − P_G − P_A − P_CS − P_GC` |

No fitted β. No mode/median μ. No Opta in this screen.

## Pool & seasons

- GW **8–38**, four **closed** seasons only (2022-23…2025-26)
- Eligible: **minutes > 60** at scored GW; **≥ 3** prior apps as-of that GW; FD AH+OU priced
- **2026/27 holdout frozen** — not in the comparison

## Metric & gates

1. As-of audit (`src/eval/gates.py`) must pass.
2. Primary: mean log predictive density per player-GW; paired **treatment − baseline** by GW; `cluster_interval` across four closed seasons.
3. **Promote if** paired interval **lo > 0**. Else keep Gaussian-total evaluation; mean layer unchanged either way.
4. `write_gated_report` only.

## Sanity (only if not clearly_lose)

DEF/GKP `pick_xi` tie-break using `P(CS)` and `P(⌊GC/2⌋≥1)` vs xp-only; realized XI points. **Not** the gate.

## Explicitly out of scope

- Replacing μ with mode/median boards  
- Retuning \(\alpha_G/\alpha_A\), shares, venue, `LAM_SHRINK` / `AWAY_SHRINK`  
- Generative MC-mean as μ (viability B1 — parked)  
- Objective-function search over event weights  
- Captain / chips  

## Deliverables

| Artifact | Path |
| --- | --- |
| Runner | `src/eval/m_discrete_event_likelihood_screen.py` |
| Gated report | `reports/m_discrete_event_likelihood_screen.md` |
| State | `REPORT_STATE.json` |
| Park / note | `.cursor/ideas.md` / `.cursor/context.md` after verdict |

## Reproduce (after implement)

```bash
PYTHONPATH=. python3 -m src.eval.m_discrete_event_likelihood_screen
```

## Decision tree

```
run gated screen
├─ clearly_wins (lo>0) → keep mean board; adopt factored likelihood for eval /
│                        optional DEF sanity rule; Opta mirror as follow-up
├─ inconclusive (covers 0) → leave Gaussian-total; park factored eval
└─ clearly_loses → park; mean board stays; discreteness ≠ better log score here
```
