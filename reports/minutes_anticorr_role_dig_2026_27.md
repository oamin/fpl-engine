# Within-season anti-corr roles → blurb → minutes rating (2026-27)

As-of only (`GW < t`). **No cross-season history.** Live seasons must fit roles from the current season's completed GWs only.

## Method

- Prior window: expanding from GW1 through `t-1` (min 5 GWs).
- Anti-corr: within-club Pearson on minutes, keep ρ ≤ -0.35.
- Roles: `nailed_starter`, `tactical_starter` (starts often, median | start in [50,75]), `impact_sub`, `rotate_pair`, else expanding default.
- Rating: tactical starters use `P(start)·median|start + P(bench)·8′`; nailed uses max(expand, 0.95·med|start); others keep expanding mean.
- Anti-corr pairs restricted to the **same position** (no GK↔outfield).
- Within-season only (no prior-season bleed into live).

## Next-GW minutes accuracy

- GWs scored: 6–9 (1099 player-GW rows).
- MAE expand (all): **24.04′**
- MAE role rating (all): **24.09′**
- Δ (expand − rating): **-0.05′** (positive ⇒ rating better).

### Tactical starters only

- n=173; mean actual=44.75722543352601; mean expand=49.96146435452794; mean rating=48.41953619598129
- MAE expand **29.20′** vs rating **29.78′** (Δ=-0.58′).

### By role

| role | n | MAE expand | MAE rating | Δ |
| --- | ---: | ---: | ---: | ---: |
| `expanding_default` | 9 | 30.74 | 30.74 | 0.00 |
| `impact_sub` | 18 | 32.01 | 32.13 | -0.12 |
| `nailed_starter` | 725 | 19.91 | 19.85 | 0.06 |
| `rotate_pair` | 174 | 34.95 | 34.95 | 0.00 |
| `tactical_starter` | 173 | 29.20 | 29.78 | -0.58 |

## Anti-corr replacement check

- Rule: when A ≤30' next GW, anti-corr B ≥45'
- Events: 299; partner ≥45′: 196; hit rate **65.6%**

## Example blurbs (as-of GW9)

- **Barkley** (Aston Villa): Often starts (50% as-of) and plays ~75' — tactical starter, not a full-90 nailed profile; anti-correlated with Manzambi (ρ=-0.98) — when one sits, the other tends to take minutes.
  - rating=41.5′ expand=44.2′ role=`tactical_starter`
- **Manzambi** (Aston Villa): Often starts (50% as-of) and plays ~71' — tactical starter, not a full-90 nailed profile; anti-correlated with Barkley (ρ=-0.98) — when one sits, the other tends to take minutes.
  - rating=39.5′ expand=39.9′ role=`tactical_starter`
- **Lavia** (Chelsea): Often starts (50% as-of) and plays ~66' — tactical starter, not a full-90 nailed profile; anti-correlated with Henderson (ρ=-0.98) — when one sits, the other tends to take minutes.
  - rating=36.8′ expand=34.6′ role=`tactical_starter`
- **George Hemmings** (Aston Villa): Often starts (50% as-of) and plays ~72' — tactical starter, not a full-90 nailed profile; anti-correlated with Manzambi (ρ=-0.97) — when one sits, the other tends to take minutes.
  - rating=40.0′ expand=37.1′ role=`tactical_starter`
- **N.Angulo** (Sunderland): Often starts (88% as-of) and plays ~64' — tactical starter, not a full-90 nailed profile; anti-correlated with Ahoka (ρ=-0.96) — when one sits, the other tends to take minutes.
  - rating=57.0′ expand=57.9′ role=`tactical_starter`
- **Hato** (Chelsea): Often starts (75% as-of) and plays ~64' — tactical starter, not a full-90 nailed profile; anti-correlated with Chavarria (ρ=-0.94) — when one sits, the other tends to take minutes.
  - rating=50.0′ expand=51.5′ role=`tactical_starter`
- **Lewis-Skelly** (Arsenal): Often starts (75% as-of) and plays ~66' — tactical starter, not a full-90 nailed profile; anti-correlated with Bruno G. (ρ=-0.93) — when one sits, the other tends to take minutes.
  - rating=51.9′ expand=50.0′ role=`tactical_starter`
- **Thomas** (Hull City): Often starts (38% as-of) and plays ~67' — tactical starter, not a full-90 nailed profile; anti-correlated with Slater (ρ=-0.93) — when one sits, the other tends to take minutes.
  - rating=30.1′ expand=33.9′ role=`tactical_starter`
- **Christie** (Bournemouth): Often starts (38% as-of) and plays ~74' — tactical starter, not a full-90 nailed profile; anti-correlated with Tavernier (ρ=-0.92) — when one sits, the other tends to take minutes.
  - rating=32.8′ expand=34.9′ role=`tactical_starter`
- **Kluivert** (Bournemouth): Often starts (62% as-of) and plays ~74' — tactical starter, not a full-90 nailed profile; anti-correlated with Christie (ρ=-0.91) — when one sits, the other tends to take minutes.
  - rating=49.2′ expand=48.0′ role=`tactical_starter`
- **Mendy** (Hull City): Often starts (62% as-of) and plays ~63' — tactical starter, not a full-90 nailed profile; anti-correlated with Herrington (ρ=-0.90) — when one sits, the other tends to take minutes.
  - rating=42.4′ expand=42.4′ role=`tactical_starter`
- **Aaronson** (Leeds): Often starts (38% as-of) and plays ~64' — tactical starter, not a full-90 nailed profile; anti-correlated with Okafor (ρ=-0.90) — when one sits, the other tends to take minutes.
  - rating=29.0′ expand=34.6′ role=`tactical_starter`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.minutes_anticorr_role_dig \
  --season 2026-27 --season-csv data/scratch/player_gw_2026_27_through_gw09.csv \
  --gw-from 6 --gw-to 9 \
  --example-as-of 9 --min-prior-gws 5
```
