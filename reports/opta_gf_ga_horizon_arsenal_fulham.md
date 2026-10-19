# Long-horizon match pots from Opta GF/GA — Arsenal vs Fulham

Pipeline: **Opta predicted rank** → season GF/GA with ±MAE bands (closed-season OLS) → Dixon–Coles match λ → standard ``xp_on_pot`` with bootstrap player shares.

- Opta as-of: **2026-10-12**
- Arsenal rank **2**, Fulham rank **20**
- Unpriced pots emitted for GW15–26 keys: **240**

## GW18: Fulham vs Arsenal

Kickoff: `2026-12-30T19:30:00Z`

| side | Opta rk | Season GF | Season GA | λ (match) | λ band | P(CS) |
|---|---:|---:|---:|---:|---|---:|
| Fulham | 20 | 35 [27,42] | 74 [66,82] | 0.70 | [0.46,0.99] | 0.08 |
| Arsenal | 2 | 75 [68,83] | 40 [33,47] | 2.42 | [1.95,2.95] | 0.45 |

Implied scoreline centre ≈ **0.70–2.42** (e_total=3.13).

### Top Fulham forecast_xp (appeared)

| player | pos | share_xG | xmi | forecast_xp |
|---|---|---:|---:|---:|
| King | MID | 0.194 | 85 | 3.05 |
| Gonzalo | FWD | 0.293 | 85 | 3.01 |
| Bobb | MID | 0.068 | 69 | 2.76 |
| Iwobi | MID | 0.049 | 82 | 2.69 |
| Palacios | MID | 0.123 | 75 | 2.64 |
| Leno | GKP | 0.000 | 90 | 2.60 |
| Berge | MID | 0.010 | 87 | 2.25 |
| Kevin | MID | 0.009 | 61 | 2.22 |

### Top Arsenal forecast_xp (appeared)

| player | pos | share_xG | xmi | forecast_xp |
|---|---|---:|---:|---:|
| Saka | MID | 0.368 | 83 | 8.73 |
| Calafiori | DEF | 0.077 | 83 | 5.66 |
| Ødegaard | MID | 0.123 | 75 | 5.58 |
| Gabriel | DEF | 0.058 | 90 | 4.92 |
| Havertz | FWD | 0.191 | 87 | 4.40 |
| White | DEF | 0.021 | 79 | 4.38 |
| Konsa | DEF | 0.026 | 90 | 4.20 |
| Raya | GKP | 0.000 | 90 | 3.91 |

## GW26: Arsenal vs Fulham

Kickoff: `2027-02-20T15:00:00Z`

| side | Opta rk | Season GF | Season GA | λ (match) | λ band | P(CS) |
|---|---:|---:|---:|---:|---|---:|
| Arsenal | 2 | 75 [68,83] | 40 [33,47] | 2.83 | [2.28,3.44] | 0.49 |
| Fulham | 20 | 35 [27,42] | 74 [66,82] | 0.60 | [0.40,0.85] | 0.05 |

Implied scoreline centre ≈ **2.83–0.60** (e_total=3.43).

### Top Arsenal forecast_xp (appeared)

| player | pos | share_xG | xmi | forecast_xp |
|---|---|---:|---:|---:|
| Saka | MID | 0.368 | 83 | 9.81 |
| Calafiori | DEF | 0.077 | 83 | 6.23 |
| Ødegaard | MID | 0.123 | 75 | 6.14 |
| Gabriel | DEF | 0.058 | 90 | 5.37 |
| Havertz | FWD | 0.191 | 87 | 4.83 |
| White | DEF | 0.021 | 79 | 4.74 |
| Konsa | DEF | 0.026 | 90 | 4.53 |
| Raya | GKP | 0.000 | 90 | 4.10 |

### Top Fulham forecast_xp (appeared)

| player | pos | share_xG | xmi | forecast_xp |
|---|---|---:|---:|---:|
| King | MID | 0.194 | 85 | 2.88 |
| Gonzalo | FWD | 0.293 | 85 | 2.84 |
| Bobb | MID | 0.068 | 69 | 2.64 |
| Iwobi | MID | 0.049 | 82 | 2.58 |
| Leno | GKP | 0.000 | 90 | 2.55 |
| Palacios | MID | 0.123 | 75 | 2.54 |
| Berge | MID | 0.010 | 87 | 2.20 |
| Kevin | MID | 0.009 | 61 | 2.18 |

## Integration

```python
from src.models.opta_gf_ga_horizon import forecast_pots_from_opta_gf_ga
pots = forecast_pots_from_opta_gf_ga(
    fixtures, team_names, start=gw, end=gw+H, priced_weeks=priced,
)
# then compute_player_forecast / make_forecast_steps as today
```

Priced weeks keep opening/Betfair pots; Opta GF/GA fills the unpriced horizon (same slot as `forecast_pots_from_outrights`).
