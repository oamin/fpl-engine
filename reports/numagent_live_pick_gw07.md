# Numagent 2.0 live pick — GW7

**PI locked** 2026-10-17 → [`data/predictions/2026-27/gw07/numagent_lock.json`](../data/predictions/2026-27/gw07/numagent_lock.json) (C Haaland, VC Mbeumo; bench Verbruggen → Murillo → Khalaili → Mainoo). Model: `score_xp_v2` + promoted `share_se_pos_prior`, L4 depth-2 γ=1, **props off**.

Primary solver: **L4**. Team `ojaminFC` (entry 2632584). Bank 5 tenths, FT 1. Odds: `snapshot` (20 priced events; t=10/10, t+1=10/10).

α_G=0.9601, α_A=0.9050. Week-t minutes: xmi `667` / rolling3 `0`. Week-t+1 minutes: rolling3 only (L4 as-of law).

## Recommendation (L4)

- Transfers: **1** (hits 0, −0 pts)
- Sold: Davis
- Bought: Affengruber
- Captain: **Haaland**
- Formation: 1-3-4-3
- Week-0 objective (μ − hits): 60.96 (XI 53.65 + C 7.31)
- Horizon objective (γ=1.0, H=2): **123.05** (week μ [53.651037983998286, 55.744875870817495] + C [7.310729791526466, 6.341540423945579])
- Bank after: 0 · FT after: 1

### XI

| Player | Pos | Club | μ | C |
| --- | --- | --- | ---: | --- |
| Raya | GKP | Arsenal | 4.43 |  |
| Rogers | MID | Chelsea | 4.71 |  |
| Thomas | DEF | Coventry City | 3.74 |  |
| Rudoni | MID | Coventry City | 3.44 |  |
| Barry | FWD | Everton | 4.43 |  |
| Affengruber | DEF | Fulham | 5.62 |  |
| Haaland | FWD | Man City | 7.31 | C |
| B.Fernandes | MID | Man Utd | 4.99 |  |
| Mbeumo | MID | Man Utd | 5.15 |  |
| Hall | DEF | Newcastle | 5.08 |  |
| Wissa | FWD | Newcastle | 4.76 |  |

### Bench

| Player | Pos | μ |
| --- | --- | ---: |
| Verbruggen | GKP | 4.41 |
| Khalaili | DEF | 2.77 |
| Mainoo | MID | 1.97 |
| Murillo | DEF | 3.30 |

## L3 contrast (week t only)

- Transfers: 1 (Murillo → Guéhi)
- Captain: Haaland
- Objective: 61.09

## Fixtures scored (week t)

- Everton v Chelsea: props 0/0
- Brentford v Liverpool: props 0/0
- Fulham v Hull City: props 0/0
- Manchester City v Ipswich Town: props 0/0
- Newcastle United v Aston Villa: props 0/0
- Bournemouth v Sunderland: props 0/0
- Brighton and Hove Albion v Crystal Palace: props 0/0
- Leeds United v Manchester United: props 0/0
- Nottingham Forest v Arsenal: props 0/0
- Tottenham Hotspur v Coventry City: props 0/0

## Fixtures scored (week t+1)

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

## Notes

- `mu_post` ≡ `score_xp_v2` (no shrink).
- L4 = BOOKIE_DEPTH=2 (t and t+1), γ=1, hard-stop, no terminal.
- Shares: 2026/27 only, GW < decision week; α pooled prior+live.
- Props: off (AH+OU+BTTS pots only).
- Week t minutes: xmi probe then rolling3; t+1: rolling3 only.
