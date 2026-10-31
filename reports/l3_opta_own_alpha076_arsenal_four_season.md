# Arsenal Opta own-λ ×0.76 — four closed seasons (GW8–38)

Do-this-1: after `match_prediction`, Arsenal-side λ × 0.76; opp λ fixed; rebuild `M`.

**Decision (PI 2026-10-30): keep α=1 as default.** Fixed 0.76 is not
season-universal (helps 24–26; worsens under-print on 22–24). No change to
`LAM_SHRINK` / `AWAY_SHRINK` / blind Opta path.

| Season | Σ FD G | Σ Opta G | Σ ×0.76 G | act G | Σ FD CS | Σ Opta CS | Σ ×0.76 CS | act CS | Σ FD GC | Σ Opta GC | Σ ×0.76 GC | act GC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 61.2 | 67.8 | 51.8 | 74 | 11.7 | 14.1 | 14.1 | 12 | 33.9 | 26.9 | 26.9 | 36 |
| 2023-24 | 68.5 | 64.5 | 49.3 | 76 | 12.3 | 12.2 | 12.2 | 15 | 30.0 | 29.7 | 29.7 | 23 |
| 2024-25 | 59.8 | 69.2 | 52.9 | 54 | 12.4 | 12.7 | 12.7 | 10 | 29.3 | 28.5 | 28.5 | 28 |
| 2025-26 | 61.9 | 76.7 | 58.8 | 57 | 13.5 | 12.8 | 12.8 | 15 | 26.6 | 28.1 | 28.1 | 24 |

Per-season plots: `l3_opta_own_alpha076_arsenal_{season}_gw8-38.png`.
