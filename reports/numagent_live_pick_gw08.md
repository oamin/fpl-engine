# Numagent 2.0 live pick — GW8

Primary solver: **L3**. Team `ojaminFC` (entry 2632584). Bank 0 tenths, FT 1. Odds: `snapshot` (10 priced events; t=10/10, t+1=0/10).

α_G=0.9601, α_A=0.9050. Week-t minutes: xmi `668` / rolling3 `0`. Week-t+1 minutes: rolling3 only (L4 as-of law).

## Recommendation (L3)

- Transfers: **1** (hits 0, −0 pts)
- Sold: Mainoo
- Bought: Lewis-Potter
- Captain: **Haaland**
- Formation: 1-4-4-2
- Week-0 objective (μ − hits): 63.74 (XI 56.57 + C 7.17)
- Bank after: 0 · FT after: 1

### XI

| Player | Pos | Club | μ | C |
| --- | --- | --- | ---: | --- |
| Raya | GKP | Arsenal | 5.07 |  |
| Lewis-Potter | MID | Brentford | 5.63 |  |
| Rogers | MID | Chelsea | 4.79 |  |
| Thomas | DEF | Coventry City | 4.62 |  |
| Affengruber | DEF | Fulham | 4.24 |  |
| Haaland | FWD | Man City | 7.17 | C |
| B.Fernandes | MID | Man Utd | 5.28 |  |
| Mbeumo | MID | Man Utd | 5.44 |  |
| Hall | DEF | Newcastle | 5.23 |  |
| Wissa | FWD | Newcastle | 4.58 |  |
| Murillo | DEF | Nott'm Forest | 4.52 |  |

### Bench

| Player | Pos | μ |
| --- | --- | ---: |
| Verbruggen | GKP | 2.88 |
| Rudoni | MID | 3.94 |
| Khalaili | DEF | 3.80 |
| Barry | FWD | 3.55 |

## Fixtures scored (week t)

- Ipswich Town v Nottingham Forest: props 0/0
- Aston Villa v Manchester City: props 0/0
- Arsenal v Everton: props 0/0
- Coventry City v Fulham: props 0/0
- Chelsea v Tottenham Hotspur: props 0/0
- Manchester United v Bournemouth: props 0/0
- Hull City v Brentford: props 0/0
- Liverpool v Brighton and Hove Albion: props 0/0
- Crystal Palace v Newcastle United: props 0/0
- Sunderland v Leeds United: props 0/0

## Fixtures scored (week t+1)


## Notes

- `mu_post` ≡ `score_xp_v2` (no shrink).
- L4 = BOOKIE_DEPTH=2 (t and t+1), γ=1, hard-stop, no terminal.
- Shares: 2026/27 only, GW < decision week; α pooled prior+live.
- Props: off (AH+OU+BTTS pots only).
- Week t minutes: xmi probe then rolling3; t+1: rolling3 only.
