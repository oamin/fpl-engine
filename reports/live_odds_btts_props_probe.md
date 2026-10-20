# Live odds probe — BTTS in M + anytime props

Fetched: `2026-10-16T13:43:33.229174+00:00` · region=`us`
Credits: before=`416` after=`409` est=`7` last_odds=`4+3` last_props=`None`
Reason: `live`

## Match markets → Dixon–Coles M (BTTS weight 1 when present)

| Match | days | AH | OU pt | BTTS? | λH | λA | ρ | xG_h | xG_a | SSE |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brentford v Liverpool | 1.01 | -1.5 | 2.5 | yes | 1.395 | 1.514 | -0.250 | 1.39 | 1.51 | 0.0028 |
| Fulham v Hull City | 1.01 | -2.5 | 2.5 | yes | 1.658 | 1.207 | +0.099 | 1.66 | 1.21 | 0.0000 |

## Anytime scorer props → share rescale demo

### Brentford v Liverpool

Players with Yes quotes: **46**

| Player | p_anytime | books |
| --- | ---: | ---: |
| Igor Thiago Nascimento Rodrigues | 0.460 | 6 |
| Callum Wilson | 0.380 | 3 |
| Alexander Isak | 0.367 | 4 |
| Kevin Schade | 0.304 | 4 |
| Lewis Koumas | 0.301 | 6 |
| Jayden Danns | 0.288 | 2 |
| Cody Gakpo | 0.279 | 4 |
| Bradley Barcola | 0.268 | 6 |
| Jaidon Anthony | 0.265 | 6 |
| Fabio Carvalho | 0.263 | 3 |
| Federico Chiesa | 0.256 | 2 |
| Gustavo Nunes Fernandes Gomes | 0.251 | 3 |

Rescale demo on home pot G≈1.34 (covered=1.34, residual=0.00):

| Player | s0 | E[G]0 | p | s1 | E[G]1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Igor Thiago Nascimento Rodrigues | 0.189 | 0.254 | 0.460 | 0.220 | 0.294 |
| Callum Wilson | 0.189 | 0.254 | 0.380 | 0.170 | 0.228 |
| Alexander Isak | 0.189 | 0.254 | 0.367 | 0.163 | 0.218 |
| Kevin Schade | 0.189 | 0.254 | 0.304 | 0.129 | 0.173 |
| Lewis Koumas | 0.189 | 0.254 | 0.301 | 0.127 | 0.171 |
| RollingReturner | 0.053 | 0.070 | 0.414 | 0.191 | 0.255 |

## Notes

- Historical FD screens keep BTTS weight 0; this live path turns it on when priced.
- Prop rescale: P(anytime)→λ=−log(1−p); pin covered E[G]; residual pot by baseline shares.
- OU functional in the fitter is still P(total≥3); non-2.5 lines are approximate until generalised.
