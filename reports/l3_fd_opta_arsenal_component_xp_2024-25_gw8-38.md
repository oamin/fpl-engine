# Arsenal 2024-25 — FD/Opta → component xP vs actuals (GW8–38)

Club Dixon–Coles moments → `score_team_side` → Σ Arsenal `xp_goals` / `xp_cs` / `xp_gc`.
Actuals = Σ Arsenal FPL P(G)/P(CS)/P(GC).

**GC (official):** `xgc = E[⌊GC/2⌋]` from `team_moments`; `xp_gc = −1 · xgc`
(matches `goals_conceded_points` / −1 per two goals).

![plot](l3_fd_opta_arsenal_component_xp_2024-25_gw8-38.png)

| | Σ FD xP | Σ Opta xP | Σ actual P | MAE FD | MAE Opta |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 278.0 | 321.6 | 264 | 5.53 | 6.49 |
| CS | 276.6 | 281.5 | 223 | 9.25 | 9.71 |
| GC | -61.8 | -58.8 | -38 | 2.09 | 2.01 |
