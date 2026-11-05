# Within-season anti-corr roles → blurb → minutes rating (2025-26)

As-of only (`GW < t`). **No cross-season history.** Live seasons must fit roles from the current season's completed GWs only.

## Method

- Prior window: expanding from GW1 through `t-1` (min 8 GWs).
- Anti-corr: within-club Pearson on minutes, keep ρ ≤ -0.35.
- Roles: `nailed_starter`, `tactical_starter` (starts often, median | start in [50,75]), `impact_sub`, `rotate_pair`, else expanding default.
- Rating: tactical starters use `P(start)·median|start + P(bench)·8′`; nailed uses max(expand, 0.95·med|start); others keep expanding mean.
- Anti-corr pairs restricted to the **same position** (no GK↔outfield).
- Live seasons must fit from **current-season GWs only** (no 2025-26 → 2026-27 bleed).

## Next-GW minutes accuracy

- GWs scored: 10–37 (10178 player-GW rows).
- MAE expand (all): **29.77′**
- MAE role rating (all): **29.45′**
- Δ (expand − rating): **0.32′** (positive ⇒ rating better).

### Tactical starters only

- n=988; mean actual=37.6751012145749; mean expand=42.05327501904302; mean rating=40.09530414670857
- MAE expand **31.65′** vs rating **31.60′** (Δ=0.05′).

### By role

| role | n | MAE expand | MAE rating | Δ |
| --- | ---: | ---: | ---: | ---: |
| `expanding_default` | 901 | 31.11 | 31.11 | 0.00 |
| `fringe` | 364 | 23.45 | 23.45 | 0.00 |
| `impact_sub` | 831 | 30.72 | 31.20 | -0.48 |
| `nailed_starter` | 5213 | 27.27 | 26.57 | 0.70 |
| `rotate_pair` | 1881 | 35.88 | 35.88 | 0.00 |
| `tactical_starter` | 988 | 31.65 | 31.60 | 0.05 |

## Anti-corr replacement check

- Rule: when A ≤30' next GW, anti-corr B ≥45'
- Events: 3436; partner ≥45′: 2288; hit rate **66.6%**

## Example blurbs (as-of GW20 tactical starters with a partner)

- **Norberto Bercique Gomes Betuncal** (Everton): Often starts (42% as-of) and plays ~72' — tactical starter, not a full-90 nailed profile; anti-correlated with Thierno Barry (ρ=-1.00) — when one sits, the other tends to take minutes.
  - rating=34.7′ expand=40.8′ role=`tactical_starter`
- **Thierno Barry** (Everton): Often starts (58% as-of) and plays ~67' — tactical starter, not a full-90 nailed profile; anti-correlated with Norberto Bercique Gomes Betuncal (ρ=-1.00) — when one sits, the other tends to take minutes.
  - rating=42.2′ expand=48.5′ role=`tactical_starter`
- **Josh King** (Fulham): Often starts (58% as-of) and plays ~66' — tactical starter, not a full-90 nailed profile; anti-correlated with Emile Smith Rowe (ρ=-0.90) — when one sits, the other tends to take minutes.
  - rating=41.6′ expand=45.4′ role=`tactical_starter`
- **Chris Wood** (Nott'm Forest): Often starts (37% as-of) and plays ~74' — tactical starter, not a full-90 nailed profile; anti-correlated with Igor Jesus Maciel da Cruz (ρ=-0.89) — when one sits, the other tends to take minutes.
  - rating=32.3′ expand=28.3′ role=`tactical_starter`
- **Jacob Bruun Larsen** (Burnley): Often starts (42% as-of) and plays ~70' — tactical starter, not a full-90 nailed profile; anti-correlated with Loum Tchaouna (ρ=-0.87) — when one sits, the other tends to take minutes.
  - rating=34.1′ expand=42.1′ role=`tactical_starter`
- **Loum Tchaouna** (Burnley): Often starts (47% as-of) and plays ~66' — tactical starter, not a full-90 nailed profile; anti-correlated with Jacob Bruun Larsen (ρ=-0.87) — when one sits, the other tends to take minutes.
  - rating=35.5′ expand=38.2′ role=`tactical_starter`
- **Anthony Elanga** (Newcastle): Often starts (37% as-of) and plays ~70' — tactical starter, not a full-90 nailed profile; anti-correlated with Jacob Murphy (ρ=-0.83) — when one sits, the other tends to take minutes.
  - rating=30.8′ expand=35.5′ role=`tactical_starter`
- **Jacob Murphy** (Newcastle): Often starts (58% as-of) and plays ~67' — tactical starter, not a full-90 nailed profile; anti-correlated with Anthony Elanga (ρ=-0.83) — when one sits, the other tends to take minutes.
  - rating=42.2′ expand=47.8′ role=`tactical_starter`
- **John McGinn** (Aston Villa): Often starts (95% as-of) and plays ~74' — tactical starter, not a full-90 nailed profile; anti-correlated with Jadon Sancho (ρ=-0.69) — when one sits, the other tends to take minutes.
  - rating=70.1′ expand=74.7′ role=`tactical_starter`
- **Bertrand Traoré** (Sunderland): Often starts (47% as-of) and plays ~63' — tactical starter, not a full-90 nailed profile; anti-correlated with Simon Adingra (ρ=-0.66) — when one sits, the other tends to take minutes.
  - rating=34.1′ expand=31.9′ role=`tactical_starter`
- **Simon Adingra** (Sunderland): Often starts (37% as-of) and plays ~68' — tactical starter, not a full-90 nailed profile; anti-correlated with Bertrand Traoré (ρ=-0.66) — when one sits, the other tends to take minutes.
  - rating=30.1′ expand=28.1′ role=`tactical_starter`
- **Emiliano Buendía Stati** (Aston Villa): Often starts (42% as-of) and plays ~58' — tactical starter, not a full-90 nailed profile; anti-correlated with Amadou Onana (ρ=-0.64) — when one sits, the other tends to take minutes.
  - rating=28.8′ expand=35.7′ role=`tactical_starter`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.minutes_anticorr_role_dig
```
