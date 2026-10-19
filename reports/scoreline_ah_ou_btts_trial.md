# Scoreline matrix trial (AH / OU / BTTS)

Generated: 2026-10-12 20:35 UTC.

Fits a Dixon–Coles grid so market functionals match fair AH cover, P(over 2.5), and P(BTTS). No neutral shrinkage.

## Synthetic battery

| Case | AH line | λH | λA | ρ | SSE | Residuals (AH/OU/BTTS) | ok |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| even_low | 0.0 | 1.180 | 1.180 | -0.000 | 0.00000 | AH -0.000 / OU -0.000 / BTTS +0.000 | yes |
| even_high | 0.0 | 1.477 | 1.477 | +0.250 | 0.00048 | AH -0.000 / OU -0.014 / BTTS +0.017 | yes |
| home_fav_05 | -0.5 | 1.777 | 0.979 | +0.169 | 0.00000 | AH +0.000 / OU -0.000 / BTTS +0.000 | yes |
| home_big_fav | -1.0 | 2.161 | 0.878 | +0.250 | 0.00051 | AH +0.010 / OU -0.014 / BTTS +0.014 | yes |
| home_quarter | -0.25 | 1.537 | 1.137 | +0.193 | 0.00000 | AH -0.000 / OU -0.000 / BTTS -0.000 | yes |
| away_fav_05 | 0.5 | 0.869 | 1.686 | -0.150 | 0.00000 | AH +0.000 / OU +0.000 / BTTS -0.000 | yes |
| under_no_btts | -0.5 | 1.569 | 0.568 | -0.250 | 0.00028 | AH -0.007 / OU +0.010 / BTTS -0.011 | yes |
| over_btts | 0.0 | 1.587 | 1.587 | +0.250 | 0.00007 | AH -0.000 / OU -0.005 / BTTS +0.006 | yes |
| stress_high_btts_low_ou | -0.5 | 1.631 | 0.898 | -0.250 | 0.01063 | AH -0.033 / OU +0.064 / BTTS -0.074 | no |
| stress_low_btts_high_ou | -0.5 | 1.776 | 0.981 | +0.250 | 0.01621 | AH +0.038 / OU -0.080 / BTTS +0.092 | no |

Synthetic hits (SSE residual gate in fitter `success`): **8/10**.

## Live US Odds API (optional)

Meta: reason=`live`, remaining=`425`, last_cost=`3`.

| Match | days | AH line | books AH/OU/BTTS | λH | λA | ρ | SSE | Residuals |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Tottenham Hotspur v Coventry City | 6.93 | -1.5 | 2/5/6 | 2.139 | 0.930 | -0.249 | 0.00000 | AH -0.000 / OU +0.000 / BTTS -0.000 |
| Nottingham Forest v Arsenal | 5.79 | 0.5 | 1/8/6 | 0.916 | 1.872 | +0.186 | 0.00000 | AH +0.000 / OU -0.000 / BTTS -0.000 |
| Bournemouth v Sunderland | 5.68 | -0.5 | 5/7/6 | 1.648 | 1.139 | -0.189 | 0.00000 | AH -0.000 / OU +0.000 / BTTS -0.000 |
| Brighton and Hove Albion v Crystal Palace | 5.68 | -1.5 | 2/4/6 | 2.081 | 1.086 | -0.035 | 0.00000 | AH +0.000 / OU -0.000 / BTTS +0.000 |
| Leeds United v Manchester United | 5.68 | -1.5 | 1/4/6 | 1.396 | 1.672 | -0.250 | 0.00000 | AH +0.000 / OU +0.000 / BTTS -0.000 |
| Ipswich Town v Nottingham Forest | 10.93 | 0.5 | 1/3/4 | 1.329 | 1.485 | -0.195 | 0.00000 | AH -0.000 / OU +0.000 / BTTS -0.000 |
| Newcastle United v Aston Villa | 4.83 | -0.5 | 1/4/6 | 1.743 | 1.377 | -0.200 | 0.00000 | AH +0.000 / OU +0.000 / BTTS -0.000 |
| Brentford v Liverpool | 4.73 | -1.5 | 1/1/6 | 1.294 | 1.310 | -0.250 | 0.01862 | AH +0.002 / OU +0.082 / BTTS -0.109 |
| Fulham v Hull City | 4.73 | -1.5 | 1/8/6 | 1.865 | 1.050 | -0.057 | 0.00000 | AH +0.000 / OU -0.000 / BTTS +0.000 |
| Manchester City v Ipswich Town | 4.73 |  |  |  |  |  |  | missing_ah_ou25_or_btts |

## Read

A small SSE means the three fair probs are jointly feasible under Dixon–Coles. Stress cases that force high BTTS with low OU (or the reverse) are the ones that should strain ρ and show larger residuals.
