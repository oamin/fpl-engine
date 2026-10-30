# Arsenal 2023-24 — FD/Opta → component xP vs actuals (GW8–38)

Club E[G]/E[CS]/E[GC] from Dixon–Coles `M` → `score_team_side` → Σ Arsenal
`xp_goals` / `xp_cs` / `xp_gc`. Actuals = Σ Arsenal FPL P(G)/P(CS)/P(GC).

![plot](l3_fd_opta_arsenal_component_xp_2023-24_gw8-38.png)

| | Σ FD xP | Σ Opta xP | Σ actual P | MAE FD | MAE Opta |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 317.8 | 298.9 | 361 | 6.58 | 6.46 |
| CS | 205.0 | 202.4 | 379 | 12.82 | 12.69 |
| GC | -204.1 | -201.3 | -35 | 5.86 | 5.70 |

Note: mean-layer `xp_gc = −1 · xGC` (linear); official P(GC) is `−⌊GC/2⌋` for GKP/DEF.
