# Scout context rotation tune trial — GW9

Trial of the five tune points: XI→id resolve, club match-quality gate, GK slot guard, rest/cup blurb veto, single vs sustained, plus precision vs actual GW9 minutes.

## Counts

| | before | after |
| --- | ---: | ---: |
| overlays | 124 | 114 |
| rotation labels | 36 | 25 |

### Context breakdown

| context | before | after |
| --- | ---: | ---: |
| `doubt` | 30 | 30 |
| `injured_expected_back` | 4 | 2 |
| `injured_unknown_return` | 50 | 53 |
| `suspected_single_week_rotation` | 36 | 25 |
| `suspended` | 4 | 4 |

## Precision vs actual minutes (≤30′ = hit)

- **Before** single-week: n=36, low(≤30′)=22, precision=61%
- **After** single-week: n=25, low(≤30′)=20, precision=80%
- **Before** sustained: n=0
- **After** sustained: n=0

### After false positives (played >30′)

| player | club | actual′ | expand′ |
| --- | --- | ---: | ---: |
| Lavia | Chelsea | 90 | 52 |
| Cherki | Man City | 35 | 60 |
| Aina | Nott'm Forest | 45 | 61 |
| Gray | Spurs | 45 | 61 |
| Yirenkyi | Coventry | 90 | 53 |

## Removed by tune (were rotation, now not)

- A.Becker (Liverpool)
- Enzo (Man City)
- Gabriel (Arsenal)
- Gonzalo (Fulham)
- Konsa (Arsenal)
- Raya (Arsenal)
- Rúben (Man City)
- Scherpen (Ipswich)
- Thomas-Asante (Coventry)
- Virgil (Liverpool)
- Yeremy (Crystal Palace)

## Remaining rotation labels (after)

| player | club | label | m0 | expand | recent |
| --- | --- | --- | ---: | ---: | --- |
| Botman | Newcastle | suspected single-week rotation | 15.0 | 90.0 | [90.0, 90.0, 90.0] |
| Guéhi | Man City | suspected single-week rotation | 15.0 | 90.0 | [90.0, 90.0, 90.0] |
| Thiaw | Newcastle | suspected single-week rotation | 15.0 | 90.0 | [90.0, 90.0, 90.0] |
| Tielemans | Man Utd | suspected single-week rotation | 15.0 | 80.4 | [90.0, 66.0, 90.0] |
| Buendía | Aston Villa | suspected single-week rotation | 15.0 | 80.2 | [74.0, 62.0, 90.0] |
| Meunier | Sunderland | suspected single-week rotation | 15.0 | 79.2 | [63.0, 75.0, 78.0] |
| Greaves | Ipswich | suspected single-week rotation | 15.0 | 71.4 | [90.0, 87.0, 0.0] |
| Barkley | Aston Villa | suspected single-week rotation | 15.0 | 61.6 | [90.0, 62.0, 6.0] |
| Gray | Spurs | suspected single-week rotation | 15.0 | 61.4 | [0.0, 84.0, 71.0] |
| Aina | Nott'm Forest | suspected single-week rotation | 15.0 | 61.2 | [58.0, 90.0, 0.0] |
| Cherki | Man City | suspected single-week rotation | 15.0 | 60.4 | [65.0, 45.0, 84.0] |
| George Hemmings | Aston Villa | suspected single-week rotation | 15.0 | 59.4 | [74.0, 45.0, 18.0] |
| Maitland-Niles | Everton | suspected single-week rotation | 15.0 | 59.0 | [13.0, 74.0, 90.0] |
| Damsgaard | Brentford | suspected single-week rotation | 15.0 | 56.6 | [85.0, 30.0, 86.0] |
| Gusto | Chelsea | suspected single-week rotation | 15.0 | 54.4 | [30.0, 77.0, 0.0] |
| Lewis-Skelly | Arsenal | suspected single-week rotation | 15.0 | 54.2 | [66.0, 45.0, 0.0] |
| Danso | Sunderland | suspected single-week rotation | 15.0 | 54.0 | [90.0, 90.0, 90.0] |
| Anthony | Brentford | suspected single-week rotation | 15.0 | 53.4 | [45.0, 59.0, 80.0] |
| Yirenkyi | Coventry | suspected single-week rotation | 15.0 | 53.0 | [27.0, 17.0, 90.0] |
| Mings | Aston Villa | suspected single-week rotation | 15.0 | 52.2 | [90.0, 65.0, 90.0] |
| Lavia | Chelsea | suspected single-week rotation | 15.0 | 51.6 | [59.0, 68.0, 0.0] |
| Schuster | Brentford | suspected single-week rotation | 15.0 | 50.6 | [73.0, 90.0, 90.0] |
| Sávio | Spurs | suspected single-week rotation | 15.0 | 48.0 | [60.0, 90.0, 90.0] |
| Yalcouyé | Brighton | suspected single-week rotation | 15.0 | 47.2 | [90.0, 64.0, 5.0] |
| Palacios | Fulham | suspected single-week rotation | 15.0 | 45.2 | [60.0, 0.0, 19.0] |

## XI match quality (after)

| club | matched | ok | unmatched sample |
| --- | --- | --- | --- |
| Arsenal | 11/11 | True | — |
| Aston Villa | 11/11 | True | — |
| Brighton | 11/11 | True | — |
| Bournemouth | 11/11 | True | — |
| Brentford | 11/11 | True | — |
| Chelsea | 11/11 | True | — |
| Coventry | 11/11 | True | — |
| Crystal Palace | 11/11 | True | — |
| Everton | 11/11 | True | — |
| Fulham | 11/11 | True | — |
| Hull | 11/11 | True | — |
| Ipswich | 11/11 | True | — |
| Leeds | 11/11 | True | — |
| Liverpool | 11/11 | True | — |
| Man City | 11/11 | True | — |
| Man Utd | 11/11 | True | — |
| Newcastle | 11/11 | True | — |
| Nott'm Forest | 11/11 | True | — |
| Sunderland | 11/11 | True | — |
| Spurs | 11/11 | True | — |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.scout_context_tune_trial \
  --gw 9 \
  --before data/predictions/2026-27/gw09/scout_minutes_context_before_tune.json \
  --ffs data/scratch/ffscout_gw09/ffscout_team_news.json \
  --bootstrap data/scratch/bootstrap_full.json \
  --live-event data/scratch/event9_live.json
```
