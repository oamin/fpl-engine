# Numagent 2.0 live pick — GW8

Primary solver: **L4**. Team `ojaminFC` (entry 2632584). Bank 0 tenths, FT 1. Odds: `live` (20 priced events; t=10/10, t+1=10/10).

α_G=0.9601, α_A=0.9050. Week-t minutes: xmi `668` / rolling3 `0`. Week-t+1 minutes: rolling3 only (L4 as-of law).

## Recommendation (L4)

- Transfers: **1** (hits 0, −0 pts)
- Sold: Mainoo
- Bought: Lewis-Potter
- Captain: **Haaland**
- Formation: 1-4-4-2
- Week-0 objective (μ − hits): 67.33 (XI 58.95 + C 8.39)
- Horizon objective (γ=1.0, H=2): **125.84** (week μ [58.947168073359926, 51.77347713618527] + C [8.386462202139128, 6.728750802700866])
- Bank after: 0 · FT after: 1

### XI

| Player | Pos | Club | μ | C |
| --- | --- | --- | ---: | --- |
| Raya | GKP | Arsenal | 4.79 |  |
| Lewis-Potter | MID | Brentford | 5.55 |  |
| Rogers | MID | Chelsea | 5.19 |  |
| Thomas | DEF | Coventry City | 4.44 |  |
| Affengruber | DEF | Fulham | 4.35 |  |
| Haaland | FWD | Man City | 8.39 | C |
| B.Fernandes | MID | Man Utd | 5.64 |  |
| Mbeumo | MID | Man Utd | 5.80 |  |
| Hall | DEF | Newcastle | 5.52 |  |
| Wissa | FWD | Newcastle | 4.82 |  |
| Murillo | DEF | Nott'm Forest | 4.47 |  |

### Bench

| Player | Pos | μ |
| --- | --- | ---: |
| Verbruggen | GKP | 2.79 |
| Rudoni | MID | 3.83 |
| Khalaili | DEF | 4.03 |
| Barry | FWD | 3.83 |

## L3 contrast (week t only)

- Transfers: 1 (Mainoo → M.Sangaré)
- Captain: Haaland
- Objective: 67.47

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

- Chelsea v Manchester United: props 0/0
- Bournemouth v Leeds United: props 0/0
- Brentford v Nottingham Forest: props 0/0
- Manchester City v Brighton and Hove Albion: props 0/0
- Coventry City v Sunderland: props 0/0
- Hull City v Ipswich Town: props 0/0
- Tottenham Hotspur v Crystal Palace: props 0/0
- Aston Villa v Fulham: props 0/0
- Liverpool v Arsenal: props 0/0
- Newcastle United v Everton: props 0/0

## Notes

- `mu_post` ≡ `score_xp_v2` (no shrink).
- L4 = BOOKIE_DEPTH=2 (t and t+1), γ=1, hard-stop, no terminal.
- Shares: 2026/27 only, GW < decision week; α pooled prior+live.
- Props: off (AH+OU+BTTS pots only).
- Week t minutes: xmi probe then rolling3; t+1: rolling3 only.
