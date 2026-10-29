# FD λ vs cascade_long Opta ranking λ

Pre-registered **2026-10-29** · protocol `cascade_long_fd_vs_opta_lam_compare` · season **2025-26** GW8–38 · wall 3.9s · n=618 paired team-sides

## Verdict

- Paired n=618: Opta−FD mean +0.068, MAE 0.306, corr 0.706.
- vs actual: Opta bias +0.122 MAE 0.918; FD bias +0.054 MAE 0.896.
- Venue split: Opta away hot / FD closer on away; FD often warmer at home — same anatomy as G-overprint isolate.
- Default venue_mode=blind unchanged; diagnostic only.

## Overall

| lens | n | mean Opta | mean FD | Opta−FD | MAE(Opta,FD) | corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| agreement | 618 | 1.517 | 1.449 | +0.068 | 0.306 | 0.706 |

| arm | bias vs act | MAE | corr(λ,act) |
| --- | ---: | ---: | ---: |
| Opta | +0.122 | 0.918 | 0.264 |
| FD | +0.054 | 0.896 | 0.271 |

## By venue

| ha | Opta−FD | Opta bias | FD bias | Opta MAE | FD MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H | -0.097 | -0.011 | +0.086 | 0.947 | 0.939 |
| A | +0.232 | +0.255 | +0.023 | 0.889 | 0.854 |

## By favourite (Opta λ)

| side | n | Opta−FD | Opta bias | FD bias |
| --- | ---: | ---: | ---: | ---: |
| favourite | 309 | +0.231 | +0.270 | +0.039 |
| underdog | 309 | -0.095 | -0.026 | +0.069 |

## By finishing-rank tercile

| band | n | Opta−FD | Opta bias | FD bias |
| --- | ---: | ---: | ---: | ---: |
| top | 217 | +0.288 | +0.326 | +0.039 |
| mid | 215 | -0.055 | -0.007 | +0.048 |
| bottom | 186 | -0.047 | +0.032 | +0.080 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_fd_vs_opta_lam_compare --season 2025-26
```

