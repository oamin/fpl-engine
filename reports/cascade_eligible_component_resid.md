# Eligible-pool cascade_short / cascade_long vs L3 — E[G, CS, GC]

Season **2025-26** · GW8–38 (31 weeks) · N=40 · goal_weight=`rate_minutes` · n=5892 player-GWs · wall 577.5s

Pre-registered: `protocol.cascade_eligible_component_resid`.

## Design

- **Pool (generate):** full-club share pool `m>0` as-of before each decision GW.
- **Pool (residual):** actual `minutes>60`, `n_prior≥3`, scored by all three arms.
- **cascade_short:** FD AH+OU centreline λ → generative MC-mean components.
- **cascade_long:** Opta place GF/GA (`venue_mode=blind`) → same generative stack.
- **L3:** as-of `score_xp_v2` centreline `xp_goals` / `xp_cs` / `xp_gc`.
- Stepped: as-of refresh each GW; decision-GW only.
- Defaults **unchanged** (diagnostic).

![bias](/opt/cursor/artifacts/cascade_eligible_component_resid_bias.png)

## Overall residual vs actual (player-GW)

| arm | n | G bias | G MAE | G corr | CS bias | CS MAE | CS corr | GC bias | GC MAE | GC corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 5892 | -0.054 | 0.859 | +0.211 | -0.094 | 0.799 | +0.292 | -0.023 | 0.241 | +0.512 |
| long | 5892 | -0.029 | 0.876 | +0.208 | -0.111 | 0.793 | +0.285 | -0.041 | 0.243 | +0.515 |
| l3 | 5892 | -0.129 | 0.814 | +0.248 | -0.184 | 0.749 | +0.304 | -0.024 | 0.243 | +0.514 |

## Quality vs L3 (paired Δ on same rows)

ΔMAE = MAE(arm) − MAE(L3); negative means arm closer to actual than L3.

| arm | G ΔMAE | G Δbias | CS ΔMAE | CS Δbias | GC ΔMAE | GC Δbias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short−L3 | +0.045 | +0.074 | +0.051 | +0.090 | -0.002 | +0.001 |
| long−L3 | +0.062 | +0.099 | +0.044 | +0.073 | +0.000 | -0.017 |

## Block totals (Σ pred − Σ actual bias implied)

| block | n | short G bias | long G bias | L3 G bias | short CS bias | long CS bias | L3 CS bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GW8-12 | 957 | -0.053 | -0.031 | -0.129 | -0.127 | -0.152 | -0.209 |
| GW13-19 | 1338 | -0.118 | -0.082 | -0.152 | -0.064 | -0.128 | -0.144 |
| GW20-26 | 1329 | -0.067 | -0.023 | -0.160 | -0.026 | -0.049 | -0.123 |
| GW27-33 | 1362 | +0.041 | +0.061 | -0.058 | -0.191 | -0.195 | -0.292 |
| GW34-38 | 906 | -0.085 | -0.094 | -0.153 | -0.057 | -0.008 | -0.147 |

## Verdict

- G MAE ranking (lower better): **l3** (0.814) < **short** (0.859) < **long** (0.876).
- CS MAE ranking: **l3** (0.749) < **long** (0.793) < **short** (0.799).
- short−L3 G ΔMAE **+0.045**; long−L3 G ΔMAE **+0.062**.
- Not a promote; defaults unchanged.

## Reproduce

```
PYTHONPATH=. python3 -m src.eval.cascade_eligible_component_resid --n 40 --goal-weight rate_minutes
```

