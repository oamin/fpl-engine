# Numagent 2.0 live pick — GW9

Primary solver: **L4**. Team `ojaminFC` (entry 2632584). Bank 0 tenths, FT 1. Odds: `snapshot` (20 priced events; t=10/10, t+1=10/10).

α_G=0.9601, α_A=0.9050. Week-t minutes: xmi `669` / rolling3 `0`. Week-t+1 minutes: rolling3 only (L4 as-of law).

## Recommendation (L4)

- Transfers: **11** (hits 10, −40 pts)
- Sold: Raya, Rudoni, Barry, Rogers, B.Fernandes, Mbeumo, Wissa, Murillo, Khalaili, Affengruber, Lewis-Potter
- Bought: Groß, Ødegaard, Tarkowski, Belloumi, Stach, Calvert-Lewin, Isak, Trafford, Gvardiol, Gabriel, Scott
- Captain: **Haaland**
- Formation: 1-4-3-3
- Week-0 objective (μ − hits): 22.68 (XI 56.03 + C 6.66)
- Horizon objective (γ=1.0, H=12): **800.36** (week μ [56.02541396266592, 64.56845106081569, 63.289152193803915, 60.958745645086225, 59.83072416741055, 65.13209559573016, 62.08193696772449, 66.9968205197009, 65.8456127849623, 64.60518425838642, 65.38553717693179, 59.83072416741055] + C [6.657443710670478, 7.349802922664222, 8.20927354642772, 6.301269696243003, 6.793250745488242, 6.861082331968697, 6.960584272110211, 7.451384771357778, 6.932763584163928, 7.063012496270922, 8.437050627933292, 6.793250745488243])
- Bank after: 4 · FT after: 1

### XI

| Player | Pos | Club | μ | C |
| --- | --- | --- | ---: | --- |
| Gabriel | DEF | Arsenal | 4.99 |  |
| Ødegaard | MID | Arsenal | 4.94 |  |
| Groß | MID | Brighton | 4.79 |  |
| Tarkowski | DEF | Everton | 5.75 |  |
| Belloumi | MID | Hull City | 4.84 |  |
| Calvert-Lewin | FWD | Leeds | 4.47 |  |
| Trafford | GKP | Leeds | 4.63 |  |
| Isak | FWD | Liverpool | 5.56 |  |
| Gvardiol | DEF | Man City | 4.74 |  |
| Haaland | FWD | Man City | 6.66 | C |
| Hall | DEF | Newcastle | 4.65 |  |

### Bench

| Player | Pos | μ |
| --- | --- | ---: |
| Scott | MID | 0.00 |
| Verbruggen | GKP | 2.72 |
| Thomas | DEF | 1.06 |
| Stach | MID | 1.90 |

## L3 contrast (week t only)

- Transfers: 1 (Khalaili → Van Hecke)
- Captain: Haaland
- Objective: 58.00

## Fixtures scored (week t)

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

## Fixtures scored (week t+1)

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

## Notes

- `mu_post` ≡ `score_xp_v2` (no shrink).
- L4 centreline horizon H=12 (`protocol.l3_centreline_forecast`): priced t:t+1 + Opta blind [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; γ=1; no terminal.
- Shares: 2026/27 only, GW < decision week; α pooled prior+live.
- Props: off (AH+OU+BTTS pots only).
- Week t minutes: xmi probe then rolling3; t+1 / Opta far: rolling3 / frozen side.
- Live near book: Odds API AH+OU+BTTS (t and t+1); far = Opta blind.
