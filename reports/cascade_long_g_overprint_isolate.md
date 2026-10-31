# Isolate cascade_long G overprint

2025-26 GW8–38 · shared L3 XI · π∝r · wall 7.0s

## Verdict

Two stacked causes — **not** one:

1. **Opta λ vs FD λ (~+18.5 of +32.5 XI resid G, ~57%)** — venue-blind place GF/GA make **away λ too high** (bias A ≈ +0.26…+0.30; home ≈ 0). Fixed `away_shrink` / `lam_shrink` parked (OOS fail).
2. **Generative remainder (~+14.0, ~43%)** — same π∝r stack on **FD** λ still sits **+14** vs actual. This is shared with cascade_short, not Opta-specific.

Structural Opta issue: place `mean_gf`/`mean_ga` are **season totals** (home+away blended). λ is then formed with `AWAY_SHRINK=1`, so the model does not know away sides score less. Actual home scoring ≈ Opta λ_H; actual away scoring ≪ Opta λ_A.

## Layer split (XI Σ casc − P on G)

| layer | resid G | share of Opta resid |
| --- | ---: | ---: |
| Opta − actual | +32.5 | 100% |
| Opta − FD (λ source) | +18.5 | 57% |
| FD − actual (generative / π∝r) | +14.0 | 43% |

## Club λ by venue

| venue | actual | Opta λ | FD λ | Opta−act | FD−act | place×venue−act |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H | 1.52 | 1.51 | 1.61 | -0.01 | +0.10 | +0.28 |
| A | 1.26 | 1.52 | 1.29 | +0.26 | +0.02 | -0.01 |

Place×venue diagnostic uses closed-season home/away scoring rates by finishing place (LOS 2025-26). It is a **mechanism check**, not a locked candidate.

## Away sides: favourite vs underdog (Opta λ − actual)

| | n | Opta λ | actual | bias |
| --- | ---: | ---: | ---: | ---: |
| away_favourite | 157 | 1.93 | 1.50 | +0.43 |
| away_underdog | 153 | 1.10 | 1.02 | +0.08 |

## XI resid_G by position (Opta π∝r)

| pos | Σ resid_G | n |
| --- | ---: | ---: |
| FWD | +30.5 | 87 |
| DEF | +3.9 | 115 |
| MID | -1.9 | 108 |
| GKP | +0.0 | 31 |

## Next steps (ordered)

1. **Do not** lock fixed `away_shrink` / `lam_shrink` (already OOS-failed).
2. **Pre-register place×venue GF/GA** (or as-of expanding home/away rates) as the Opta λ map — mechanism matches the H≈0 / A≈+0.30 split; screen club λ OOS on 2022-23…2024-25 before any default flip.
3. **Separately** dig the FD/generative **+14** under π∝r (and confirm under default `rate_minutes`) — minutes gate, α_G empty mass, or share concentration. That is cascade_short’s problem too.
4. Keep decision metric = player-GW likelihood; XI resid is diagnostic.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_long_g_overprint_isolate
```

