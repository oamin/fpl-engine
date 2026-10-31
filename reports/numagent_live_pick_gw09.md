# Numagent 2.0 live pick — GW9

**PI locked** 2026-10-31 → [`data/predictions/2026-27/gw09/numagent_lock.json`](../data/predictions/2026-27/gw09/numagent_lock.json) (C Haaland, VC Isak; bench Verbruggen → Thomas → Rudoni → Khalaili).

Deadline **2026-10-31T11:00:00Z**. Primary solver: **L4 priced H=2** (Odds t:t+1 only — **no Opta far**). Team `ojaminFC` (entry 2632584). Bank 0 tenths, FT 1. Odds: snapshot (20 priced; t=10/10, t+1=10/10; credits remaining 111). Live hit cap: **max_hits=1**.

α_G=0.9601, α_A=0.9050. Week-t minutes: deep whitelist packets (Guardian + BBC / Coventry Telegraph / Chronicle / Sports Mole / Standard / Football London / club + PL) + FFScout XIs → `gw09_xmi_probe.csv` (xmi `669`). Week-t+1: rolling3 (L4 as-of).

## Recommendation (L4)

- Transfers: **2** (hits 1, −4 pts)
- Sold: B.Fernandes, Wissa
- Bought: Ødegaard, Isak
- Captain: **Haaland** · suggested vice: **Isak**
- Formation: 1-3-4-3
- Week-0 objective (μ − hits): 53.79 (XI 51.22 + C 6.57)
- Horizon objective (γ=1.0, H=2): **119.03**
- Bank after: 19 · FT after: 1

### Team-news notes (owned / moves)

- **Thomas** (bench): FPL 50% calf; Lampard late call (BBC / Coventry Telegraph / PL) → xmi **50/50** (~40′); FFS XI has Mfuni.
- **B.Fernandes** out: Carrick clear-the-air; captain / will start vs Chelsea (BBC / Sports Mole / FFS) — H=2 still sells for Isak/Ødegaard path.
- **Wissa** out: Osula ankle doubt (BBC / Chronicle); FFS/Chronicle project Wissa lone striker vs Everton — horizon still sells.
- **Isak** in: Standard + PL Bloom / FFS Liverpool XI vs Arsenal (xmi ~82).
- **Ødegaard** in: FFS Arsenal XI (Saka/Saliba out).
- **Rogers / Hall / Haaland / Mbeumo**: firm XI per FFS + local desk (Football London Rogers; NUFC Hall).
- Prior Opta H=12 diagnostic (Fernandes+Wissa → Groß+Isak) archived under `numagent_pick_h12_uncapped_hits.json` / `reports/numagent_live_pick_gw09_h12_uncapped_hits.md` — **not** the deadline plan.

### XI

| Player | Pos | Club | μ | C |
| --- | --- | ---: | ---: | --- |
| Raya | GKP | Arsenal | 3.81 |  |
| Ødegaard | MID | Arsenal | 4.89 |  |
| Lewis-Potter | MID | Brentford | 4.34 |  |
| Rogers | MID | Chelsea | 4.77 |  |
| Barry | FWD | Everton | 4.72 |  |
| Affengruber | DEF | Fulham | 4.33 |  |
| Isak | FWD | Liverpool | 5.81 |  |
| Haaland | FWD | Man City | 6.57 | C |
| Mbeumo | MID | Man Utd | 4.18 |  |
| Hall | DEF | Newcastle | 4.60 |  |
| Murillo | DEF | Nott'm Forest | 3.20 |  |

### Bench

| Player | Pos | μ |
| --- | --- | ---: |
| Verbruggen | GKP | 2.72 |
| Thomas | DEF | 1.07 |
| Rudoni | MID | 3.24 |
| Khalaili | DEF | 2.66 |

## L3 contrast (week t only)

- Transfers: 1 (Murillo → Van Hecke)
- Captain: Haaland
- Objective: 57.51

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
- L4 centreline horizon H=2 (`protocol.l3_centreline_forecast`): priced t:t+1 only; **no Opta far**; γ=1; no terminal.
- Shares: 2026/27 only, GW < decision week; α pooled prior+live.
- Props: off (AH+OU+BTTS pots only).
- Week t minutes: xmi probe then rolling3; t+1: rolling3 (as-of).
- Live near book: Odds API AH+OU+BTTS (t and t+1); no Opta far.
