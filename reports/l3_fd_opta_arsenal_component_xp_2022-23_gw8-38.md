# Arsenal 2022-23 — FD/Opta → component xP vs actuals (GW8–38)

Club E[G]/E[CS]/E[GC] from Dixon–Coles `M` → `score_team_side` → Σ Arsenal
`xp_goals` / `xp_cs` / `xp_gc`. Actuals = Σ Arsenal FPL P(G)/P(CS)/P(GC).

![plot](l3_fd_opta_arsenal_component_xp_2022-23_gw8-38.png)

| | Σ FD xP | Σ Opta xP | Σ actual P | MAE FD | MAE Opta |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 281.4 | 311.7 | 351 | 5.96 | 6.16 |
| CS | 249.5 | 301.6 | 298 | 12.17 | 12.39 |
| GC | -241.2 | -191.0 | -73 | 5.84 | 4.39 |

Note: mean-layer `xp_gc = −1 · xGC` (linear); official P(GC) is `−⌊GC/2⌋` for GKP/DEF.
