# Raw player stats ↔ FPL points

Two questions:
1. **Same-match** — given what happened, how tightly do stats map to points?
2. **Predictive** — do prior rates forecast *next* match points?

## Same-match multivariate (decomposition)

| scope | features | n | R² |
|---|---|---:|---:|
| ALL | `minutes,goals,assists,clean_sheets,bonus,bps,yellow_cards,red_cards` | 11498 | 0.913 |
| GKP | `minutes,goals,assists,clean_sheets,saves,bonus,bps,goals_conceded,yellow_cards` | 767 | 0.937 |
| DEF | `minutes,goals,assists,clean_sheets,bonus,bps,goals_conceded,defcon,yellow_cards` | 3950 | 0.961 |
| MID | `minutes,goals,assists,clean_sheets,bonus,bps,defcon,yellow_cards` | 5347 | 0.968 |
| FWD | `minutes,goals,assists,bonus,bps,yellow_cards` | 1434 | 0.992 |
| MID | `goals,assists,bonus` | 5347 | 0.857 |
| FWD | `goals,assists,bonus` | 1434 | 0.965 |

## Same-match univariate — top features by scope

### ALL

| feature | n | corr | R² |
|---|---:|---:|---:|
| bps | 11498 | 0.904 | 0.818 |
| bonus | 11498 | 0.767 | 0.588 |
| goals | 11498 | 0.652 | 0.426 |
| clean_sheets | 11498 | 0.480 | 0.230 |
| minutes | 11498 | 0.394 | 0.155 |
| xG | 11498 | 0.385 | 0.148 |

### GKP

| feature | n | corr | R² |
|---|---:|---:|---:|
| bps | 767 | 0.913 | 0.833 |
| clean_sheets | 767 | 0.860 | 0.740 |
| bonus | 767 | 0.719 | 0.517 |
| goals_conceded | 767 | -0.710 | 0.505 |
| saves | 767 | 0.242 | 0.059 |
| minutes | 767 | 0.128 | 0.016 |

### DEF

| feature | n | corr | R² |
|---|---:|---:|---:|
| bps | 3950 | 0.899 | 0.808 |
| clean_sheets | 3950 | 0.747 | 0.558 |
| bonus | 3950 | 0.702 | 0.492 |
| goals | 3950 | 0.485 | 0.235 |
| goals_conceded | 3950 | -0.416 | 0.173 |
| defcon | 3950 | 0.374 | 0.140 |

### MID

| feature | n | corr | R² |
|---|---:|---:|---:|
| bps | 5347 | 0.898 | 0.806 |
| bonus | 5347 | 0.801 | 0.641 |
| goals | 5347 | 0.773 | 0.598 |
| assists | 5347 | 0.484 | 0.235 |
| minutes | 5347 | 0.459 | 0.211 |
| xG | 5347 | 0.448 | 0.201 |

### FWD

| feature | n | corr | R² |
|---|---:|---:|---:|
| bps | 1434 | 0.959 | 0.920 |
| goals | 1434 | 0.909 | 0.826 |
| bonus | 1434 | 0.877 | 0.769 |
| xG | 1434 | 0.622 | 0.386 |
| minutes | 1434 | 0.407 | 0.165 |
| assists | 1434 | 0.376 | 0.142 |

## Predictive univariate — top prior rates by scope

Filter: minutes ≥ 60.0, prior apps ≥ 3.

### ALL

| feature | n | corr | R² |
|---|---:|---:|---:|
| prior_xGI | 6973 | 0.135 | 0.018 |
| prior_xG | 6973 | 0.121 | 0.015 |
| prior_total_points | 6973 | 0.107 | 0.011 |
| prior_goals | 6973 | 0.103 | 0.011 |
| prior_bps | 6973 | 0.099 | 0.010 |
| prior_bonus | 6973 | 0.094 | 0.009 |

### GKP

| feature | n | corr | R² |
|---|---:|---:|---:|
| prior_saves | 648 | -0.078 | 0.006 |
| prior_xA | 648 | 0.064 | 0.004 |
| prior_xGI | 648 | 0.051 | 0.003 |
| prior_clean_sheets | 648 | 0.047 | 0.002 |
| prior_total_points | 648 | 0.042 | 0.002 |
| prior_xG | 648 | -0.033 | 0.001 |

### DEF

| feature | n | corr | R² |
|---|---:|---:|---:|
| prior_bps | 2698 | 0.107 | 0.011 |
| prior_total_points | 2698 | 0.100 | 0.010 |
| prior_clean_sheets | 2698 | 0.077 | 0.006 |
| prior_xGI | 2698 | 0.067 | 0.004 |
| prior_xG | 2698 | 0.061 | 0.004 |
| prior_goals | 2698 | 0.056 | 0.003 |

### MID

| feature | n | corr | R² |
|---|---:|---:|---:|
| prior_xGI | 2936 | 0.141 | 0.020 |
| prior_xG | 2936 | 0.124 | 0.015 |
| prior_xA | 2936 | 0.109 | 0.012 |
| prior_total_points | 2936 | 0.105 | 0.011 |
| prior_bonus | 2936 | 0.087 | 0.008 |
| prior_bps | 2936 | 0.082 | 0.007 |

### FWD

| feature | n | corr | R² |
|---|---:|---:|---:|
| prior_bonus | 691 | 0.183 | 0.034 |
| prior_xGI | 691 | 0.180 | 0.032 |
| prior_xG | 691 | 0.177 | 0.031 |
| prior_total_points | 691 | 0.166 | 0.028 |
| prior_goals | 691 | 0.165 | 0.027 |
| prior_bps | 691 | 0.160 | 0.025 |

## Predictive multivariate vs points-baseline

| scope | features | n | R² | R²(prior points only) | ΔR² |
|---|---|---:|---:|---:|---:|
| ALL | `prior_total_points,prior_minutes,prior_xGI` | 6973 | 0.022 | 0.011 | 0.010 |
| GKP | `prior_total_points,prior_minutes,prior_saves,prior_clean_sheets` | 648 | 0.011 | 0.002 | 0.010 |
| DEF | `prior_total_points,prior_minutes,prior_clean_sheets,prior_defcon,prior_xGI` | 2698 | 0.015 | 0.010 | 0.005 |
| MID | `prior_total_points,prior_minutes,prior_xGI,prior_goals,prior_assists` | 2936 | 0.022 | 0.011 | 0.011 |
| FWD | `prior_total_points,prior_minutes,prior_xGI,prior_goals` | 691 | 0.044 | 0.028 | 0.016 |

## Plots

- `data/plots/raw_stats_same_match.png`
- `data/plots/raw_stats_decomp.png`
- `data/plots/raw_stats_predictive.png`

## Read

- High same-match R² ⇒ points are a near-deterministic function of events (BPS/bonus/G/A/CS).
- Low predictive R² ⇒ those events are hard to forecast; baselines beat chasing noise.
- Useful trends are channels we *can* predict (minutes, goal involvement rates), not markets→points clouds.
