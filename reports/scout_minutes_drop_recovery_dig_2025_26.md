# Scout dig — minutes drops & recovery (2025-26)

Exploratory only (not a gated promote). Question: when a regular's
minutes fall >1σ below expanding mean, how many GWs until recovery,
and does a ρ-decay toward expanding fit the return path?

## Definitions

- Pool eligibility (as-of): ≥5 prior GWs, expanding mean ≥45.0′.
- **Drop:** minutes < expand_mean − 1.0·expand_sd (sd floored at 5.0).
- **Recover:** minutes ≥ expand_mean − 0.5·expand_sd (as-of before that GW).
- **Zero drop:** minutes = 0 at drop GW; **soft:** 0 < m < 45.

## Counts

- Events: **794** (700 recovered, 94 censored at season end).
- Players with ≥1 event: **277**.
- Zero-drop share: **51.3%**; soft: **31.9%**.
- Events/player: mean 2.87; ≥2 events 73.6%.

## Recovery duration (uncensored)

- Median **2.0** GW (mean 2.87; IQR 1–4; p90 6).
- ≤1 GW: **46.4%**; ≤3 GW: **74.4%**; ≥6 GW: **13.1%**.

### By drop type

| type | n | median GW | mean | ≤1 | ≤3 | ≥6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero (m=0) | 340 | 2.0 | 3.50 | 36% | 67% | 19% |
| soft (0<m<45) | 233 | 1.0 | 2.44 | 51% | 78% | 8% |
| other (m≥45 but <μ−σ) | 127 | 1.0 | 1.98 | 67% | 89% | 6% |

### By position (uncensored)

| pos | n | median | mean | ≤3 |
| --- | ---: | ---: | ---: | ---: |
| DEF | 264 | 2.0 | 2.73 | 75% |
| FWD | 78 | 1.0 | 2.73 | 77% |
| GK | 24 | 3.0 | 5.12 | 58% |
| MID | 334 | 2.0 | 2.86 | 74% |

## Return path (unconditional mean minutes after drop)

All uncensored events, including those already recovered (avoids
survivor bias). Compared to `m(h)=expand+(m0−expand)·ρ^h`.

| h | n | actual mean | share m=0 | ρ=0.50 | ρ=0.70 | ρ=0.85 | ρ=0.95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 700 | 17.3 | 49% | 17.3 | 17.3 | 17.3 | 17.3 |
| 1 | 691 | 45.3 | 33% | 43.9 | 33.3 | 25.3 | 20.0 |
| 2 | 683 | 49.2 | 28% | 57.2 | 44.5 | 32.2 | 22.7 |
| 3 | 677 | 51.1 | 26% | 63.8 | 52.3 | 37.9 | 25.1 |
| 4 | 644 | 52.8 | 25% | 67.3 | 57.9 | 42.9 | 27.4 |
| 5 | 638 | 54.6 | 23% | 69.0 | 61.7 | 47.1 | 29.7 |
| 6 | 632 | 56.0 | 23% | 70.0 | 64.6 | 50.8 | 31.9 |
| 7 | 597 | 55.6 | 22% | 70.6 | 66.7 | 54.2 | 34.4 |
| 8 | 587 | 55.6 | 22% | 70.8 | 67.9 | 56.6 | 36.0 |

RMSE vs actual on h=1…6 (lower better):

- ρ=0.5: **40.61**′
- ρ=0.7: **39.69**′
- ρ=0.85: **41.96**′
- ρ=0.95: **48.48**′

### Mixture: quick vs long zero-drops

| bucket | n | mean m path h=0…4 |
| --- | ---: | --- |
| zero_quick_le1 | 122 | 0, 83, 70, 63, 69 |
| zero_long_ge4 | 113 | 0, 1, 2, 2, 16 |

## Long recoveries (uncensored)

| player | pos | team | drop GW | m | expand | recover GW | recover m |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Chris Wood | FWD | Nott'm Forest | 7 | 63 | 76 | 25 | 25 |
| Altay Bayındır | GK | Man Utd | 7 | 0 | 90 | 24 | 0 |
| James Ward-Prowse | MID | West Ham | 6 | 0 | 83 | 19 | 0 |
| Tyrone Mings | DEF | Aston Villa | 6 | 31 | 90 | 17 | 15 |
| Justin Kluivert | MID | Bournemouth | 21 | 0 | 46 | 16 | 14 |
| Joško Gvardiol | DEF | Man City | 21 | 0 | 65 | 15 | 57 |
| Oscar Bobb | MID | Man City | 10 | 0 | 50 | 15 | 7 |
| Mads Hermansen | GK | West Ham | 6 | 0 | 72 | 15 | 0 |

## Signals (readout)

- Median recovery **~2 GW**; almost half bounce in ≤1 GW → many drops are
  rotation/one-offs, not multi-week injuries.
- **Zero drops split:** ~1/3 back next GW, ~1/3 still out ≥4 GW. One ρ
  cannot serve both; need a duration tag (`out_for_n_gws`) or a mixture.
- Soft drops (0<m<45) recover faster (median 1) than zeros (median 2).
- Best single ρ on the unconditional return curve is a clue for the
  default decay — still a 2025-26 dig, not a four-season lock.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.scout_minutes_drop_recovery_dig
```

