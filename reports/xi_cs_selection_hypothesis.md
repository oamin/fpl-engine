# XI CS underprint — selection hypothesis test

Season **2025-26** · GW [8, 9, 10, 11, 12] · 5 seeds from `cascade_vs_l3_component_xp_multiseed.json` · bootstrap n=5000 · wall 0.4s

## Hypothesis

The multi-seed FPL XI CS gap (xP_CS ≪ actual CS pts) is **selection**: opens stack DEF/GKP whose **team-GWs** kept more sheets than FD `xcs` implied — not a broken Dixon–Coles prior (league team-GW is calibrated).

## League baseline (all team-GWs GW8–12)

- n=100  mean xCS=0.269  actual CS=0.270  bias(p−y)=-0.001

## Per seed (unique team-GWs of XI GKP+DEF with m≥60)

| seed | n team-GW | mean xCS | actual CS | bias p−y | boot p05 | p_left | player CS rate | casc Σr_CS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l3_wc | 14 | 0.414 | 0.714 | -0.300 | -0.192 | 0.004 | 0.70 | -22.7 |
| v2_no_haaland | 15 | 0.408 | 0.667 | -0.259 | -0.195 | 0.014 | 0.67 | -22.2 |
| v2_no_arsenal | 17 | 0.336 | 0.471 | -0.135 | -0.177 | 0.104 | 0.50 | -21.1 |
| v2_no_liverpool | 18 | 0.391 | 0.556 | -0.165 | -0.169 | 0.055 | 0.58 | -21.9 |
| casc_ol | 14 | 0.387 | 0.500 | -0.113 | -0.196 | 0.160 | 0.55 | -6.2 |

**Mean selected bias:** -0.194 (all negative: True). Mean bootstrap left-tail p: 0.067.

## Read

- `p_left` = fraction of size-matched league bootstrap samples with bias ≤ the seed’s bias (small ⇒ selected set is unusually sheet-lucky vs FD).
- Selected mean xCS is also **higher** than the league (~0.27): opens chase good CS fixtures, and those fixtures then **over-delivered** vs even that elevated xCS.

## Verdict

Selection hypothesis supported: selected backs' team-GWs have actual CS ≫ FD xCS while the league panel is calibrated; selected bias sits in the left tail of a size-matched bootstrap.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.xi_cs_selection_hypothesis
```

