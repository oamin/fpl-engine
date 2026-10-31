# Attack shares without minutes in the weights

Season **2025-26** · OL XI GW8–19 · full club m>0 · N=40 · wall 129.8s · no share-SE shrink

## Arms

| arm | rule |
| --- | --- |
| **comp_rm** | locked: ``w=r·m/90``, ``π=w/Σw`` |
| **comp_r** | ``π=r/Σr`` among m>0 (minutes = on-dice gate only) |

## OL XI Σ E[G] vs actual

Σ G = **28.0** · α_G=0.956

| arm | Σ E[G] | Σ resid | Δ vs comp_rm | MAE | XI π mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| comp_rm | 38.20 | +10.20 | +0.00 | 0.332 | 0.255 |
| comp_r | 28.23 | +0.22 | -9.97 | 0.302 | 0.179 |

## Top resid players (comp_rm)

| name | pos | G | E_rm | E_r | r_rm | r_r |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Yankuba Minteh | MID | 0 | 3.20 | 1.90 | +3.20 | +1.90 |
| Nick Woltemade | FWD | 1 | 3.27 | 2.08 | +2.27 | +1.08 |
| João Pedro Junqueira de Jesus | FWD | 1 | 2.77 | 1.75 | +1.77 | +0.75 |
| Kevin Schade | MID | 0 | 1.30 | 1.02 | +1.30 | +1.02 |
| Riccardo Calafiori | DEF | 0 | 0.85 | 0.75 | +0.85 | +0.75 |
| Granit Xhaka | MID | 1 | 1.83 | 0.93 | +0.83 | -0.07 |
| Omar Alderete | DEF | 0 | 0.77 | 0.75 | +0.77 | +0.75 |
| Antoine Semenyo | MID | 2 | 2.67 | 1.85 | +0.67 | -0.15 |
| Jurriën Timber | DEF | 0 | 0.33 | 0.35 | +0.33 | +0.35 |
| Gabriel dos Santos Magalhães | DEF | 1 | 1.27 | 0.65 | +0.28 | -0.35 |

## Verdict

Locked **comp_rm** (π∝r·m) XI resid **+10.2**. **comp_r** (π∝r, minutes gate only) +0.2 (Δ -10.0). Dropping minutes from share weights cuts the G overprint. Club pot still renormalises; only the weight changes.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_share_no_renorm_minutes --n 40
```

