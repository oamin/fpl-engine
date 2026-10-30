# Arsenal 2024-25 — FD/Opta → component xP vs actuals (GW8–38)

Club E[G]/E[CS]/E[GC] from Dixon–Coles `M` → `score_team_side` → Σ Arsenal
`xp_goals` / `xp_cs` / `xp_gc`. Actuals = Σ Arsenal FPL P(G)/P(CS)/P(GC).

![plot](l3_fd_opta_arsenal_component_xp_2024-25_gw8-38.png)

| | Σ FD xP | Σ Opta xP | Σ actual P | MAE FD | MAE Opta |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 278.0 | 321.6 | 264 | 5.53 | 6.49 |
| CS | 276.6 | 281.5 | 223 | 9.25 | 9.71 |
| GC | -220.9 | -214.1 | -38 | 6.16 | 5.96 |

Note: mean-layer `xp_gc = −1 · xGC` (linear); official P(GC) is `−⌊GC/2⌋` for GKP/DEF.
