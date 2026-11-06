# Numagent 2.0 live pick — GW10

Primary solver: **L4**. Team `ojaminFC` (entry 2632584). Bank 19 tenths, FT 1. Odds: `snapshot` (12 priced events; t=10/10, t+1=2/10).

α_G=0.9601, α_A=0.9050. Week-t minutes: xmi `670` / rolling3 `0`. Week-t+1 minutes: rolling3 only (L4 as-of law).

## Recommendation (L4)

- Transfers: **6** (hits 5, −20 pts)
- Sold: Thomas, Rudoni, Rogers, Murillo, Khalaili, Lewis-Potter
- Bought: Jacquet, Gakpo, Guéhi, Gvardiol, Mainoo, Charles
- Captain: **Haaland**
- Formation: 1-3-4-3
- Week-0 objective (μ − hits): 50.10 (XI 62.79 + C 7.31)
- Horizon objective (γ=1.0, H=2): **106.99** (week μ [62.79283855738505, 50.10396120384281] + C [7.311983388496628, 6.7828032735741015])
- Bank after: 3 · FT after: 1

### XI

| Player | Pos | Club | μ | C |
| --- | --- | --- | ---: | --- |
| Raya | GKP | Arsenal | 5.13 |  |
| Ødegaard | MID | Arsenal | 6.06 |  |
| Barry | FWD | Everton | 6.19 |  |
| Gakpo | MID | Liverpool | 5.69 |  |
| Isak | FWD | Liverpool | 7.27 |  |
| Guéhi | DEF | Man City | 4.62 |  |
| Gvardiol | DEF | Man City | 5.02 |  |
| Haaland | FWD | Man City | 7.31 | C |
| Mbeumo | MID | Man Utd | 5.02 |  |
| Mainoo | MID | Man Utd | 5.74 |  |
| Hall | DEF | Newcastle | 4.76 |  |

### Bench

| Player | Pos | μ |
| --- | --- | ---: |
| Verbruggen | GKP | 4.09 |
| Charles | MID | 4.90 |
| Affengruber | DEF | 4.46 |
| Jacquet | DEF | 4.42 |

## L3 contrast (week t only)

- Transfers: 1 (Khalaili → Tarkowski)
- Captain: Haaland
- Objective: 69.83

## Fixtures scored (week t)

- Everton v Coventry City: props 0/0
- Leeds United v Tottenham Hotspur: props 0/0
- Arsenal v Hull City: props 0/0
- Fulham v Newcastle United: props 0/0
- Nottingham Forest v Manchester City: props 0/0
- Ipswich Town v Bournemouth: props 0/0
- Brighton and Hove Albion v Brentford: props 0/0
- Sunderland v Chelsea: props 0/0
- Crystal Palace v Liverpool: props 0/0
- Manchester United v Aston Villa: props 0/0

## Fixtures scored (week t+1)

- Manchester City v Fulham: props 0/0
- Liverpool v Manchester United: props 0/0

## Notes

- `mu_post` ≡ `score_xp_v2` (no shrink).
- L4 = BOOKIE_DEPTH=2 (t and t+1), γ=1, hard-stop, no terminal.
- Shares: 2026/27 only, GW < decision week; α pooled prior+live.
- Props: off (AH+OU+BTTS pots only).
- Week t minutes: xmi probe then rolling3; t+1: rolling3 only.
