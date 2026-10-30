# Arsenal 2025-26 — FD/Opta → component xP vs actuals (GW8–38)

Club Dixon–Coles moments → `score_team_side` → Σ Arsenal `xp_goals` / `xp_cs` / `xp_gc`.
Actuals = Σ Arsenal FPL P(G)/P(CS)/P(GC).

**GC (official):** `xgc = E[⌊GC/2⌋]` from `team_moments`; `xp_gc = −1 · xgc`
(matches `goals_conceded_points` / −1 per two goals).

![plot](l3_fd_opta_arsenal_component_xp_2025-26_gw8-38.png)

| | Σ FD xP | Σ Opta xP | Σ actual P | MAE FD | MAE Opta |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 291.6 | 361.2 | 254 | 4.86 | 6.34 |
| CS | 262.7 | 248.6 | 349 | 11.28 | 11.38 |
| GC | -56.3 | -60.7 | -41 | 2.09 | 2.03 |
