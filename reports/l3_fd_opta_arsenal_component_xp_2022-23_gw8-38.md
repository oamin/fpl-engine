# Arsenal 2022-23 — FD/Opta → component xP vs actuals (GW8–38)

Club Dixon–Coles moments → `score_team_side` → Σ Arsenal `xp_goals` / `xp_assists` /
`xp_cs` / `xp_gc`. Actuals = Σ Arsenal FPL P(G)/P(A)/P(CS)/P(GC).

- **Assists:** `P(xA)=Σ xp_assists`; `P(A)=3×assists`.
- **GC:** `xgc=E[⌊GC/2⌋]`; `xp_gc=−1·xgc`.

![plot](l3_fd_opta_arsenal_component_xp_2022-23_gw8-38.png)

| | Σ FD xP | Σ Opta xP | Σ actual P | MAE FD | MAE Opta |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 281.4 | 311.7 | 351 | 5.96 | 6.16 |
| A | 156.9 | 173.8 | 192 | 3.20 | 3.23 |
| CS | 249.5 | 301.6 | 298 | 12.17 | 12.39 |
| GC | -71.9 | -50.0 | -73 | 2.77 | 2.61 |
