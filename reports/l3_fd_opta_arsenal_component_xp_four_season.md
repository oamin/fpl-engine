# Arsenal FD/Opta → component xP vs actuals — four closed seasons (GW8–38)

Club Dixon–Coles moments → as-of `score_team_side` → Σ Arsenal `xp_goals`/`xp_cs`/`xp_gc`
vs Σ Arsenal actual FPL P(G)/P(CS)/P(GC). α=1 Opta (identity).

| Season | Σ FD G | Σ Opta G | act G | MAE FD G | MAE Opta G | Σ FD CS | Σ Opta CS | act CS | MAE FD CS | MAE Opta CS | Σ FD GC | Σ Opta GC | act GC | MAE FD GC | MAE Opta GC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 281.4 | 311.7 | 351 | 5.96 | 6.16 | 249.5 | 301.6 | 298 | 12.17 | 12.39 | -241.2 | -191.0 | -73 | 5.84 | 4.39 |
| 2023-24 | 317.8 | 298.9 | 361 | 6.58 | 6.46 | 205.0 | 202.4 | 379 | 12.82 | 12.69 | -204.1 | -201.3 | -35 | 5.86 | 5.70 |
| 2024-25 | 278.0 | 321.6 | 264 | 5.53 | 6.49 | 276.6 | 281.5 | 223 | 9.25 | 9.71 | -220.9 | -214.1 | -38 | 6.16 | 5.96 |
| 2025-26 | 291.6 | 361.2 | 254 | 4.86 | 6.34 | 262.7 | 248.6 | 349 | 11.28 | 11.38 | -210.9 | -222.3 | -41 | 5.78 | 6.04 |

Per-season: `l3_fd_opta_arsenal_component_xp_{season}_gw8-38.png`.

Note: mean-layer `xp_gc = −1·xGC`; official P(GC) is `−⌊GC/2⌋` for GKP/DEF — GC totals are not on the same scale.
