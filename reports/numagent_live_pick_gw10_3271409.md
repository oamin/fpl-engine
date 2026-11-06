# Numagent 2.0 live pick — GW10

Primary solver: **L4**. Team `hiking w djed spence` (entry 3271409). Bank 17 tenths, FT 1. Odds: `snapshot` (12 priced events; t=10/10, t+1=2/10).

α_G=0.9601, α_A=0.9050. Week-t minutes: xmi `670` / rolling3 `0`. Week-t+1 minutes: rolling3 only (L4 as-of law).

## Recommendation (L4)

- Transfers: **6** (hits 5, −20 pts)
- Sold: De Cuyper, João Pedro, Thomas, Rogers, Barnes, Schade
- Bought: Gakpo, Isak, Guéhi, Gvardiol, Mainoo, Charles
- Captain: **Haaland**
- Formation: 1-4-4-2
- Week-0 objective (μ − hits): 51.25 (XI 63.94 + C 7.31)
- Horizon objective (γ=1.0, H=2): **103.81** (week μ [63.936417324764186, 45.77527181237006] + C [7.311983388496628, 6.7828032735741015])
- Bank after: 3 · FT after: 1

### XI

| Player | Pos | Club | μ | C |
| --- | --- | --- | ---: | --- |
| Calafiori | DEF | Arsenal | 5.91 |  |
| Verbruggen | GKP | Brighton | 4.09 |  |
| Groß | MID | Brighton | 5.68 |  |
| Tarkowski | DEF | Everton | 6.87 |  |
| Gakpo | MID | Liverpool | 5.69 |  |
| Isak | FWD | Liverpool | 7.27 |  |
| Gvardiol | DEF | Man City | 5.02 |  |
| Haaland | FWD | Man City | 7.31 | C |
| Mbeumo | MID | Man Utd | 5.02 |  |
| Mainoo | MID | Man Utd | 5.74 |  |
| Van Hecke | DEF | Spurs | 5.33 |  |

### Bench

| Player | Pos | μ |
| --- | --- | ---: |
| Gonzalo | FWD | 4.66 |
| Charles | MID | 4.90 |
| Guéhi | DEF | 4.62 |
| Kinsky | GKP | 3.52 |

## L3 contrast (week t only)

- Transfers: 1 (João Pedro → Isak)
- Captain: Haaland
- Objective: 69.51

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
