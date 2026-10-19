# Selectable pool — expanding minutes + FFScout firm

## FFScout availability

Public page: [team-news](https://www.fantasyfootballscout.co.uk/team-news).

| Available | Notes |
|---|---|
| Predicted XI (11/club) | Public HTML |
| Out / Doubts (%) / Banned | Public HTML |
| Club news blurb | Public HTML |
| Explicit `nailed_on` / `firm` labels | **Not published** — we map firm/nailed := XI − Out − Banned − Doubts |
| Members projections / Rate My Team | Login wall — not used |

Fetched: `2026-10-19T09:39:02.415482+00:00` · clubs with firm XI match: **20** · selectable players: **180** (mean minutes > 60.0, apps ≥ 2, minutes via `bootstrap_proxy`).

## Per-club firm selection

| Club | FFS firm | Matched | Selectable | Opponent |
|---|---:|---:|---:|---|
| Arsenal | 11 | 11 | 9 | Nottingham Forest (A) |
| Aston Villa | 10 | 10 | 9 | Newcastle United (A) |
| Brighton | 11 | 11 | 10 | Crystal Palace (H) |
| Bournemouth | 11 | 11 | 10 | Sunderland (H) |
| Brentford | 9 | 9 | 6 | Liverpool (H) |
| Chelsea | 9 | 9 | 7 | Everton (A) |
| Coventry | 11 | 11 | 9 | Tottenham Hotspur (A) |
| Crystal Palace | 11 | 11 | 10 | Brighton and Hove Albion (A) |
| Everton | 11 | 11 | 10 | Chelsea (H) |
| Fulham | 11 | 11 | 11 | Hull City (H) |
| Hull | 11 | 11 | 8 | Fulham (A) |
| Ipswich | 11 | 11 | 9 | Manchester City (A) |
| Leeds | 11 | 11 | 11 | Manchester United (H) |
| Liverpool | 11 | 11 | 9 | Brentford (A) |
| Man City | 11 | 11 | 11 | Ipswich Town (H) |
| Man Utd | 10 | 10 | 6 | Leeds United (A) |
| Newcastle | 11 | 11 | 9 | Aston Villa (H) |
| Nott'm Forest | 10 | 10 | 10 | Arsenal (H) |
| Sunderland | 10 | 10 | 9 | Bournemouth (A) |
| Spurs | 9 | 9 | 7 | Coventry City (H) |

## Selectable pool (head)

| id | name | team | pos | mean mins | apps |
|---:|---|---|---|---:|---:|
| 1 | Raya | Arsenal | GKP | 90 | 5 |
| 8 | Calafiori | Arsenal | DEF | 83 | 5 |
| 4 | Gabriel | Arsenal | DEF | 90 | 5 |
| 31 | Konsa | Arsenal | DEF | 94 | 3 |
| 10 | White | Arsenal | DEF | 79 | 4 |
| 13 | Rice | Arsenal | MID | 82 | 5 |
| 12 | Saka | Arsenal | MID | 83 | 5 |
| 15 | Ødegaard | Arsenal | MID | 75 | 5 |
| 26 | Havertz | Arsenal | FWD | 87 | 5 |
| 597 | Suzuki | Aston Villa | GKP | 90 | 4 |
| 32 | Cash | Aston Villa | DEF | 99 | 3 |
| 37 | Lindelöf | Aston Villa | DEF | 90 | 5 |
| 34 | Pau | Aston Villa | DEF | 62 | 3 |
| 41 | Buendía | Aston Villa | MID | 80 | 5 |
| 54 | Gomes | Aston Villa | MID | 64 | 2 |
| 47 | Kamara | Aston Villa | MID | 89 | 5 |
| 45 | McGinn | Aston Villa | MID | 87 | 5 |
| 166 | N.Jackson | Aston Villa | FWD | 72 | 4 |
| 57 | Petrović | Bournemouth | GKP | 90 | 5 |
| 60 | Hill | Bournemouth | DEF | 90 | 5 |
| 566 | Silva | Bournemouth | DEF | 90 | 5 |
| 64 | Smith | Bournemouth | DEF | 87 | 5 |
| 61 | Truffert | Bournemouth | DEF | 90 | 5 |
| 73 | Adams | Bournemouth | MID | 93 | 3 |
| 70 | Kluivert | Bournemouth | MID | 74 | 4 |
| 67 | Rayan | Bournemouth | MID | 87 | 5 |
| 68 | Tavernier | Bournemouth | MID | 85 | 5 |
| 79 | Evanilson | Bournemouth | FWD | 85 | 5 |
| 82 | Kelleher | Brentford | GKP | 90 | 5 |
| 87 | Ajer | Brentford | DEF | 90 | 5 |
| 88 | Kayode | Brentford | DEF | 74 | 4 |
| 105 | Anthony | Brentford | MID | 89 | 3 |
| 86 | Lewis-Potter | Brentford | MID | 81 | 5 |
| 106 | Thiago | Brentford | FWD | 88 | 5 |
| 109 | Verbruggen | Brighton | GKP | 90 | 5 |
| 114 | Boscagli | Brighton | DEF | 87 | 5 |
| 115 | De Cuyper | Brighton | DEF | 85 | 5 |
| 116 | Dunk | Brighton | DEF | 90 | 4 |
| 113 | F.Kadıoğlu | Brighton | DEF | 83 | 4 |
| 504 | Vuskovic | Brighton | DEF | 87 | 5 |
| … | (140 more) | | | | |

## Use

Short- and long-horizon squad selection should restrict candidates to `selectable_ids` in `selectable_pool.json`.

```bash
PYTHONPATH=. python3 -m src.models.selectable_pool
```
