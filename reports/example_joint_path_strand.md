# Example joint path strand (2025-26)

> **Superseded numbers:** generated under climb-contaminated γ=0.9 and
> concave φ·W PE. Ledger shape is still illustrative. Current PE is
> `U(b)=ν·b` with ν=`HIT_COST` (+4/FT, +20 at cap); see
> `reports/cascade_ft_pe_path.md`.

Protocol path arm · season **2025-26** · decision GW6 → GW8–19 · strand **0** (seed chain from `7`, strand_seed=1390851128) · legacy `ft_pe_nu=2`.

One coherent MC roll; joint MILP plans the opening 15 and every later transfer/hold. FT bank is next-deadline bank (`advance_ft`).

PE (legacy on this page): terminal `U(f_H)=ν·φ(f_H)·W` with ν=2, W=7.18 — **retired**. Current engine: terminal `U(f_H)=ν·f_H`, ν=4.

MILP: candidates=48, Optimization terminated successfully. (HiGHS Status 7: Optimal).

## Transfer ledger

| GW | action | free | paid | hit pts | FT before → after | ν·φ(ft_after) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | **open** (free rebuild) | 0 | 0 | 0 | 0 → 1 | 2.00 |
| 9 | −Nordi Mukiele → +Joachim Andersen | 1 | 0 | 0 | 1 → 1 | 2.00 |
| 10 | −Maxence Lacroix, Antoine Semenyo → +Kevin Schade, Lewis Dunk | 1 | 1 | 4 | 1 → 1 | 2.00 |
| 11 | −Jurriën Timber, Anton Stach, Joachim Andersen → +Nathan Collins, Granit Xhaka, Nordi Mukiele | 1 | 2 | 8 | 1 → 1 | 2.00 |
| 12 | −Richarlison de Andrade → +Igor Thiago Nascimento Rodrigues | 1 | 0 | 0 | 1 → 1 | 2.00 |
| 13 | −Igor Thiago Nascimento Rodrigues → +Ollie Watkins | 1 | 0 | 0 | 1 → 1 | 2.00 |
| 14 | −Matthijs de Ligt → +Marc Guéhi | 1 | 0 | 0 | 1 → 1 | 2.00 |
| 15 | −Ollie Watkins → +João Pedro Junqueira de Jesus | 1 | 0 | 0 | 1 → 1 | 2.00 |
| 16 | −Bruno Borges Fernandes → +Morgan Rogers | 1 | 0 | 0 | 1 → 1 | 2.00 |
| 17 | −Dan Burn → +Michael Keane | 1 | 0 | 0 | 1 → 1 | 2.00 |
| 18 | −Bryan Mbeumo → +Tijjani Reijnders | 1 | 0 | 0 | 1 → 1 | 2.00 |
| 19 | −Nordi Mukiele → +Marcos Senesi Barón | 1 | 0 | 0 | 1 → 1 | 2.00 (**U=14.4**) |

## Week-by-week squads

### GW8

Free open · `ft_after=1`

- **XI:** Jurriën Timber, Bryan Mbeumo, Yankuba Minteh, Maxence Lacroix, Erling Haaland, Bruno Borges Fernandes, Dan Burn, Richarlison de Andrade (C), Jarrod Bowen, Robin Roefs, Nordi Mukiele
- **Bench:** Antoine Semenyo, Matthijs de Ligt, Guglielmo Vicario, Anton Stach

### GW9

1 free transfer(s) · FT 1 → 1

- Out: Nordi Mukiele
- In: Joachim Andersen

- **XI:** Antoine Semenyo, Bryan Mbeumo, Maxence Lacroix, Joachim Andersen, Erling Haaland, Matthijs de Ligt, Bruno Borges Fernandes, Dan Burn, Guglielmo Vicario, Richarlison de Andrade, Jarrod Bowen (C)
- **Bench:** Jurriën Timber, Yankuba Minteh, Anton Stach, Robin Roefs

### GW10

2 transfer(s): 1 free + 1 paid (−4 pts) · FT 1 → 1

- Out: Maxence Lacroix, Antoine Semenyo
- In: Kevin Schade, Lewis Dunk

- **XI:** Jurriën Timber, Kevin Schade, Lewis Dunk, Yankuba Minteh (C), Joachim Andersen, Erling Haaland, Matthijs de Ligt, Bruno Borges Fernandes, Dan Burn, Guglielmo Vicario, Anton Stach
- **Bench:** Bryan Mbeumo, Richarlison de Andrade, Jarrod Bowen, Robin Roefs

### GW11

3 transfer(s): 1 free + 2 paid (−8 pts) · FT 1 → 1

- Out: Jurriën Timber, Anton Stach, Joachim Andersen
- In: Nathan Collins, Granit Xhaka, Nordi Mukiele

- **XI:** Nathan Collins, Bryan Mbeumo, Kevin Schade, Lewis Dunk, Yankuba Minteh (C), Erling Haaland, Richarlison de Andrade, Jarrod Bowen, Granit Xhaka, Robin Roefs, Nordi Mukiele
- **Bench:** Matthijs de Ligt, Bruno Borges Fernandes, Dan Burn, Guglielmo Vicario

### GW12

1 free transfer(s) · FT 1 → 1

- Out: Richarlison de Andrade
- In: Igor Thiago Nascimento Rodrigues

- **XI:** Kevin Schade, Igor Thiago Nascimento Rodrigues, Yankuba Minteh, Erling Haaland (C), Matthijs de Ligt, Bruno Borges Fernandes, Dan Burn, Jarrod Bowen, Granit Xhaka, Robin Roefs, Nordi Mukiele
- **Bench:** Nathan Collins, Bryan Mbeumo, Lewis Dunk, Guglielmo Vicario

### GW13

1 free transfer(s) · FT 1 → 1

- Out: Igor Thiago Nascimento Rodrigues
- In: Ollie Watkins

- **XI:** Ollie Watkins (C), Nathan Collins, Bryan Mbeumo, Kevin Schade, Lewis Dunk, Erling Haaland, Matthijs de Ligt, Dan Burn, Guglielmo Vicario, Jarrod Bowen, Granit Xhaka
- **Bench:** Yankuba Minteh, Bruno Borges Fernandes, Robin Roefs, Nordi Mukiele

### GW14

1 free transfer(s) · FT 1 → 1

- Out: Matthijs de Ligt
- In: Marc Guéhi

- **XI:** Ollie Watkins, Bryan Mbeumo, Kevin Schade, Yankuba Minteh, Marc Guéhi, Erling Haaland, Bruno Borges Fernandes (C), Dan Burn, Guglielmo Vicario, Jarrod Bowen, Nordi Mukiele
- **Bench:** Nathan Collins, Lewis Dunk, Granit Xhaka, Robin Roefs

### GW15

1 free transfer(s) · FT 1 → 1

- Out: Ollie Watkins
- In: João Pedro Junqueira de Jesus

- **XI:** Bryan Mbeumo (C), Lewis Dunk, Yankuba Minteh, João Pedro Junqueira de Jesus, Erling Haaland, Bruno Borges Fernandes, Dan Burn, Guglielmo Vicario, Jarrod Bowen, Granit Xhaka, Nordi Mukiele
- **Bench:** Nathan Collins, Kevin Schade, Marc Guéhi, Robin Roefs

### GW16

1 free transfer(s) · FT 1 → 1

- Out: Bruno Borges Fernandes
- In: Morgan Rogers

- **XI:** Morgan Rogers (C), Nathan Collins, Bryan Mbeumo, Kevin Schade, Lewis Dunk, João Pedro Junqueira de Jesus, Erling Haaland, Jarrod Bowen, Granit Xhaka, Robin Roefs, Nordi Mukiele
- **Bench:** Yankuba Minteh, Marc Guéhi, Dan Burn, Guglielmo Vicario

### GW17

1 free transfer(s) · FT 1 → 1

- Out: Dan Burn
- In: Michael Keane

- **XI:** Bryan Mbeumo, Kevin Schade, Lewis Dunk, Yankuba Minteh, João Pedro Junqueira de Jesus, Marc Guéhi, Michael Keane, Erling Haaland (C), Guglielmo Vicario, Jarrod Bowen, Granit Xhaka
- **Bench:** Morgan Rogers, Nathan Collins, Robin Roefs, Nordi Mukiele

### GW18

1 free transfer(s) · FT 1 → 1

- Out: Bryan Mbeumo
- In: Tijjani Reijnders

- **XI:** Morgan Rogers, Nathan Collins, Kevin Schade (C), Yankuba Minteh, Marc Guéhi, Tijjani Reijnders, Erling Haaland, Jarrod Bowen, Granit Xhaka, Robin Roefs, Nordi Mukiele
- **Bench:** Lewis Dunk, João Pedro Junqueira de Jesus, Michael Keane, Guglielmo Vicario

### GW19

1 free transfer(s) · FT 1 → 1

- Out: Nordi Mukiele
- In: Marcos Senesi Barón

- **XI:** Morgan Rogers, Marcos Senesi Barón, Kevin Schade, Lewis Dunk, Yankuba Minteh, João Pedro Junqueira de Jesus, Michael Keane, Tijjani Reijnders (C), Erling Haaland, Guglielmo Vicario, Jarrod Bowen
- **Bench:** Nathan Collins, Marc Guéhi, Granit Xhaka, Robin Roefs

