# XI CS selection — same clubs, other 2025-26 windows

Season **2025-26** · freeze clubs from GW8–12 XI backs · 5 seeds · bootstrap n=5000 · wall 1.4s

## Design

Take each multi-seed open’s **selected back clubs** (GKP/DEF with m≥60 in the GW8–12 panel). Hold that club set fixed. On each contiguous GW block, compare those clubs’ FD `xcs` vs actual CS against a size-matched league bootstrap in the **same** block.

This asks whether GW8–12 sheet luck was unique to that window, or whether the same selection stays lucky elsewhere in 2025-26.

## Window summary (mean across seeds)

| window | league bias | mean sel bias | all neg? | mean p_left |
| --- | ---: | ---: | --- | ---: |
| GW1-7 | -0.015 | -0.105 | True | 0.147 |
| GW8-12 | -0.001 | -0.095 | True | 0.189 |
| GW13-19 | +0.022 | -0.041 | True | 0.215 |
| GW20-26 | +0.037 | +0.060 | False | 0.612 |
| GW27-33 | -0.033 | +0.028 | False | 0.744 |
| GW34-38 | +0.030 | -0.056 | False | 0.237 |

## Per seed × window

| seed | window | n | mean xCS | actual CS | bias | p_left |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| l3_wc | GW1-7 | 28 | 0.310 | 0.429 | -0.119 | 0.102 |
| l3_wc | GW8-12 | 20 | 0.364 | 0.500 | -0.136 | 0.084 |
| l3_wc | GW13-19 | 28 | 0.352 | 0.393 | -0.041 | 0.215 |
| l3_wc | GW20-26 | 29 | 0.332 | 0.276 | +0.056 | 0.583 |
| l3_wc | GW27-33 | 27 | 0.302 | 0.259 | +0.043 | 0.808 |
| l3_wc | GW34-38 | 19 | 0.324 | 0.368 | -0.045 | 0.197 |
| v2_no_haaland | GW1-7 | 28 | 0.310 | 0.429 | -0.119 | 0.108 |
| v2_no_haaland | GW8-12 | 20 | 0.364 | 0.500 | -0.136 | 0.087 |
| v2_no_haaland | GW13-19 | 28 | 0.352 | 0.393 | -0.041 | 0.216 |
| v2_no_haaland | GW20-26 | 29 | 0.332 | 0.276 | +0.056 | 0.584 |
| v2_no_haaland | GW27-33 | 27 | 0.302 | 0.259 | +0.043 | 0.797 |
| v2_no_haaland | GW34-38 | 19 | 0.324 | 0.368 | -0.045 | 0.179 |
| v2_no_arsenal | GW1-7 | 35 | 0.269 | 0.314 | -0.046 | 0.323 |
| v2_no_arsenal | GW8-12 | 25 | 0.286 | 0.320 | -0.034 | 0.338 |
| v2_no_arsenal | GW13-19 | 35 | 0.272 | 0.343 | -0.071 | 0.098 |
| v2_no_arsenal | GW20-26 | 36 | 0.255 | 0.194 | +0.060 | 0.621 |
| v2_no_arsenal | GW27-33 | 34 | 0.242 | 0.176 | +0.065 | 0.906 |
| v2_no_arsenal | GW34-38 | 24 | 0.262 | 0.208 | +0.053 | 0.608 |
| v2_no_liverpool | GW1-7 | 28 | 0.310 | 0.429 | -0.119 | 0.107 |
| v2_no_liverpool | GW8-12 | 20 | 0.364 | 0.500 | -0.136 | 0.080 |
| v2_no_liverpool | GW13-19 | 28 | 0.352 | 0.393 | -0.041 | 0.216 |
| v2_no_liverpool | GW20-26 | 29 | 0.332 | 0.276 | +0.056 | 0.590 |
| v2_no_liverpool | GW27-33 | 27 | 0.302 | 0.259 | +0.043 | 0.811 |
| v2_no_liverpool | GW34-38 | 19 | 0.324 | 0.368 | -0.045 | 0.194 |
| casc_ol | GW1-7 | 28 | 0.270 | 0.393 | -0.123 | 0.096 |
| casc_ol | GW8-12 | 20 | 0.317 | 0.350 | -0.033 | 0.358 |
| casc_ol | GW13-19 | 28 | 0.276 | 0.286 | -0.010 | 0.329 |
| casc_ol | GW20-26 | 29 | 0.279 | 0.207 | +0.072 | 0.681 |
| casc_ol | GW27-33 | 28 | 0.268 | 0.321 | -0.054 | 0.400 |
| casc_ol | GW34-38 | 19 | 0.276 | 0.474 | -0.198 | 0.008 |

## Verdict

Sheet luck for these clubs is **early-season**, not only GW8–12: GW1–12 mean sel bias -0.100; GW13–38 mean -0.002 (near zero / flips sign). League stays ~calibrated in every block.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.xi_cs_selection_hypothesis --scan-windows
```

