# Dig FD/generative +14 XI resid G

Pre-registered **2026-10-27** · protocol `cascade_fd_generative_g_dig` · wall 6.9s

## Verdict

- π∝r (`rate`) resid G: **+14.0** (FWD +14.7).
- Default `rate_minutes` resid G: **+100.3** (FWD +46.9).
- Δ (rate − rate_minutes): **-86.3** (FWD Δ -32.2).
- Mean α_G empty mass (π∝r map): **0.041** (α_G≈0.959) — too small to explain +14.

## Position split

| pos | π∝r resid G | rate_minutes resid G | n (π∝r) |
| --- | ---: | ---: | ---: |
| GKP | +0.0 | +0.0 | 31 |
| DEF | +4.2 | +28.7 | 115 |
| MID | -4.9 | +24.8 | 108 |
| FWD | +14.7 | +46.9 | 87 |

## Share concentration probe

- **rate**: mean HHI=0.146, top1=0.252, FWD mass=0.334
- **rate_minutes**: mean HHI=0.171, top1=0.290, FWD mass=0.331
- Δ rate−rm: HHI=-0.025, top1=-0.037, FWD mass=+0.003

## Next

- Default `rate_minutes` is the hot arm on this map — do **not** treat π∝r +14 as the live generative problem.
- Share-concentration / minutes gate is the next generative lever (player-GW likelihood gate before any default flip).
- Keep Opta λ venue work separate (`venue_away` club-λ candidate).

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_fd_generative_g_dig --run-rate-minutes
```

