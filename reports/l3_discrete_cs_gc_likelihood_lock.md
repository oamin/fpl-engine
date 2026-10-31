# Lock — discrete CS/GC predictive likelihood (GKP/DEF)

**Locked:** 2026-10-30  
**Protocol:** `l3_discrete_cs_gc_likelihood` · screen `m_skel_cs_gc_likelihood_screen`  
**Status:** locked (eval default) · mean layer unchanged

## What passed

Pre-registered same-skeleton screen on four closed seasons (GW8–38, FD
centreline, GKP/DEF only):

| Arm | Law |
| --- | --- |
| Treatment | Bernoulli \(P(\mathrm{CS})\) from \(M\) + categorical \(P(\lfloor GA/2\rfloor=k)\) from \(M\) |
| Baseline | Gaussian(`xp_cs`/`xp_gc`, σ=3) |
| Shared | Gaussian G/A/rest from locked `score_xp_v2` μ |

| Metric | Value |
| --- | ---: |
| Mean Δ log / GW | **+2.8276** |
| Paired interval | **[+2.7980, +2.8579]** |
| Verdict | **promote → locked** |

Component Δ (pooled): CS **+1.66**, GC **+1.17**. Gates: as-of audit passed;
`write_gated_report` used. Report: `m_skel_cs_gc_likelihood_screen.md`.

## Lock meaning

- **Adopt** discrete CS/GC from Dixon–Coles \(M\) as the predictive-likelihood
  law for GKP/DEF CS/GC terms under a factored player-GW log score.
- **Do not** change `score_team_side` / `score_xp_v2` means, α_G/α_A, shares,
  venue shrinks, or cascade μ defaults.
- **Do not** use mode/median \(M\) boards as μ (parked earlier).
- Visual companion: `l3_fd_arsenal_discrete_cs_gc_from_M_2025-26_gw8-38.md`.

## Explicitly not locked (parked / lost)

| Arm | Verdict | Report |
| --- | --- | --- |
| Factored five-term vs single Gaussian-on-total | park_clear_lose | `m_discrete_event_likelihood_screen.md` |
| Generative MC-mean as replacement μ | clearly_loses | viability B1/B2; `cascade_short_mean_tt1_vs_gen.md` |
| Pooled same-skeleton incl. MID/FWD null terms | diagnostic only (inflated) | superseded by GKP/DEF screen |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.m_skel_cs_gc_likelihood_screen
```
