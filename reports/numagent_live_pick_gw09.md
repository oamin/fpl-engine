# Numagent 2.0 live pick — GW9

Deadline **2026-10-31T11:00:00Z**. Primary solver: **L4** centreline H=12 (Odds t:t+1 + Opta far). Team `ojaminFC` (entry 2632584). Bank 0 tenths, FT 1. Odds: live pull → snapshot (20 priced; t=10/10, t+1=10/10; credits remaining 111). Live hit cap: **max_hits=1**.

α_G=0.9601, α_A=0.9050. Week-t minutes: FFScout + Guardian packets → `gw09_xmi_probe.csv` (xmi `669`). Week-t+1 / Opta far: rolling3 / frozen side.

## Recommendation (L4)

- Transfers: **2** (hits 1, −4 pts)
- Sold: B.Fernandes, Wissa
- Bought: Groß, Isak
- Captain: **Haaland** · suggested vice: **Isak**
- Formation: 1-3-4-3
- Week-0 objective (μ − hits): 53.99 (XI 51.34 + C 6.66)
- Horizon objective (γ=1.0, H=12): **721.26**
- Bank after: 24 · FT after: 1

### Team-news notes (owned / moves)

- **Thomas** (bench): FPL 50% calf; Lampard late call → xmi 50/50 (~40′).
- **B.Fernandes** out: still starts for United per FFS/PL, but L4 horizon prefers Isak/Groß path.
- **Wissa** out: FFS XI lead vs Everton; horizon still sells for Isak.
- **Isak** in: FFS Liverpool XI; Anfield vs Arsenal.
- **Groß** in: FFS Brighton XI away at City.
- Uncapped-hits H=12 diagnostic (11 transfers / −40) archived as `numagent_pick_h12_uncapped_hits.json` — not the deadline plan.

### XI

| Player | Pos | Club | μ | C |
| --- | --- | --- | ---: | --- |
| Raya | GKP | Arsenal | 3.81 |  |
| Lewis-Potter | MID | Brentford | 4.51 |  |
| Groß | MID | Brighton | 4.79 |  |
| Rogers | MID | Chelsea | 5.00 |  |
| Barry | FWD | Everton | 4.72 |  |
| Affengruber | DEF | Fulham | 4.32 |  |
| Isak | FWD | Liverpool | 5.56 |  |
| Haaland | FWD | Man City | 6.66 | C |
| Mbeumo | MID | Man Utd | 4.11 |  |
| Hall | DEF | Newcastle | 4.65 |  |
| Murillo | DEF | Nott'm Forest | 3.21 |  |

### Bench

| Player | Pos | μ |
| --- | --- | ---: |
| Verbruggen | GKP | 2.72 |
| Thomas | DEF | 1.06 |
| Rudoni | MID | 3.21 |
| Khalaili | DEF | 2.68 |

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
