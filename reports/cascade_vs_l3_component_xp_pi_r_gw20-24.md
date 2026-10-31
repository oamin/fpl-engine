# Cascade π∝r vs L3/L4 — player xP(G, CS, GA) + actual (no captain)

Season **2025-26** · GW [20, 21, 22, 23, 24] · N=40 · goal_weight=`rate` (π∝r (minutes gate only)) · wall 367.4s

## What this is

- **Same GW8 team** for both arms: L3/L4 wildcard open (`l3_l4_gw08_seed_2025_26.json`).
- GW9–12 follow the locked **L3** transfer path; XI = `pick_xi` on `score_xp_v2` among that owned 15.
- Both scorers are read on **that same XI** (no captain, no hits):
  - **cascade** — generative mean `xp_goals` / `xp_cs` / `xp_gc` (goal shares `rate`)
  - **l3_v2** — as-of `score_xp_v2` centreline components (L4 μ_post≡v2 when mode=none; default π∝r·m)
  - **actual** — FPL `P_G` / `P_CS` / `P_GA` and full `total_points`

## XI totals by GW (no captain)

| GW | casc ΣG | casc ΣCS | casc ΣGA | l3 ΣG | l3 ΣCS | l3 ΣGA | P ΣG | P ΣCS | P ΣGA | Σ total_pts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 7.9 | 6.6 | -2.3 | 9.7 | 6.1 | -2.5 | 28.0 | 4.0 | -1.0 | 65 |
| 21 | 8.0 | 7.2 | -1.6 | 10.2 | 7.3 | -1.7 | 18.0 | 10.0 | -2.0 | 48 |
| 22 | 10.0 | 8.8 | -1.4 | 10.3 | 8.0 | -1.8 | 0.0 | 18.0 | 0.0 | 45 |
| 23 | 10.5 | 8.5 | -1.2 | 12.0 | 6.6 | -1.3 | 11.0 | 1.0 | -3.0 | 35 |
| 24 | 8.7 | 8.6 | -1.5 | 10.1 | 8.3 | -1.5 | 13.0 | 9.0 | -3.0 | 56 |

**Overall** (55 player-GWs): casc ΣG/CS/GA = 45.0/39.7/-8.0; l3 = 52.3/36.4/-8.8; actual = 70.0/42.0/-9.0; Σ total_points = 249.

| component | MAE casc | MAE l3_v2 |
| --- | ---: | ---: |
| G | 1.621 | 1.736 |
| CS | 0.921 | 0.884 |
| GA | 0.187 | 0.200 |

## Player rows

### GW20 — pick_xi_score_xp_v2 · Σ total_pts=65

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Guglielmo Vicario | GKP | 90 | 0.00 | 1.90 | -0.23 | 0.00 | 1.58 | -0.25 | 0 | 0 | 0 | 2 |
| Malick Thiaw | DEF | 90 | 1.05 | 1.50 | -0.35 | 0.38 | 1.51 | -0.27 | 6 | 4 | 0 | 17 |
| Diogo Dalot Teixeira | DEF | 90 | 0.30 | 1.00 | -0.35 | 0.16 | 1.03 | -0.45 | 0 | 0 | 0 | 2 |
| Marc Guéhi | DEF | 90 | 0.45 | 0.70 | -0.72 | 0.25 | 0.62 | -0.69 | 0 | 0 | -1 | 2 |
| Trevoh Chalobah | DEF | 90 | 0.15 | 0.60 | -0.68 | 0.38 | 0.46 | -0.83 | 0 | 0 | 0 | 4 |
| Declan Rice | MID | 90 | 0.62 | 0.33 | 0.00 | 0.96 | 0.38 | 0.00 | 10 | 0 | 0 | 17 |
| Ryan Gravenberch | MID | 90 | 0.12 | 0.38 | 0.00 | 0.72 | 0.32 | 0.00 | 0 | 0 | 0 | 2 |
| Ethan Ampadu | MID | 0 | 0.25 | 0.20 | 0.00 | 0.62 | 0.24 | 0.00 | 0 | 0 | 0 | 0 |
| Erling Haaland | FWD | 90 | 2.10 | 0.00 | 0.00 | 2.89 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Igor Thiago Nascimento Rodrigues | FWD | 90 | 1.30 | 0.00 | 0.00 | 1.81 | 0.00 | 0.00 | 12 | 0 | 0 | 17 |
| Hugo Ekitiké | FWD | 0 | 1.50 | 0.00 | 0.00 | 1.49 | 0.00 | 0.00 | 0 | 0 | 0 | 0 |

### GW21 — pick_xi_score_xp_v2 · Σ total_pts=48

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 1.20 | -0.20 | 0.00 | 1.47 | -0.28 | 0 | 4 | 0 | 6 |
| Michael Keane | DEF | 82 | 0.45 | 1.80 | -0.25 | 0.55 | 1.66 | -0.23 | 6 | 0 | 0 | 5 |
| Trevoh Chalobah | DEF | 90 | 0.45 | 1.10 | -0.45 | 0.48 | 1.17 | -0.39 | 0 | 0 | -1 | 3 |
| Diogo Dalot Teixeira | DEF | 90 | 0.30 | 1.80 | -0.38 | 0.22 | 1.41 | -0.30 | 0 | 0 | -1 | 1 |
| Marc Guéhi | DEF | 90 | 0.00 | 0.70 | -0.33 | 0.31 | 0.94 | -0.49 | 0 | 4 | 0 | 6 |
| Declan Rice | MID | 90 | 0.25 | 0.30 | 0.00 | 0.92 | 0.37 | 0.00 | 0 | 1 | 0 | 3 |
| Ryan Gravenberch | MID | 90 | 0.62 | 0.10 | 0.00 | 0.44 | 0.15 | 0.00 | 0 | 1 | 0 | 3 |
| Ethan Ampadu | MID | 90 | 0.00 | 0.17 | 0.00 | 0.42 | 0.16 | 0.00 | 0 | 0 | 0 | 3 |
| Erling Haaland | FWD | 90 | 3.00 | 0.00 | 0.00 | 3.38 | 0.00 | 0.00 | 4 | 0 | 0 | 6 |
| Igor Thiago Nascimento Rodrigues | FWD | 81 | 2.30 | 0.00 | 0.00 | 2.66 | 0.00 | 0.00 | 8 | 0 | 0 | 12 |
| Hugo Ekitiké | FWD | 0 | 0.60 | 0.00 | 0.00 | 0.86 | 0.00 | 0.00 | 0 | 0 | 0 | 0 |

### GW22 — pick_xi_score_xp_v2 · Σ total_pts=45

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.20 | -0.07 | 0.00 | 1.78 | -0.21 | 0 | 4 | 0 | 6 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.45 | 2.20 | -0.07 | 0.37 | 1.78 | -0.21 | 0 | 4 | 0 | 8 |
| Trevoh Chalobah | DEF | 90 | 0.45 | 1.30 | -0.28 | 0.58 | 1.33 | -0.33 | 0 | 4 | 0 | 8 |
| Marc Guéhi | DEF | 0 | 0.00 | 1.20 | -0.33 | 0.30 | 1.23 | -0.36 | 0 | 0 | 0 | 0 |
| Diogo Dalot Teixeira | DEF | 90 | 0.75 | 0.60 | -0.65 | 0.17 | 0.58 | -0.72 | 0 | 4 | 0 | 5 |
| Declan Rice | MID | 90 | 0.25 | 0.55 | 0.00 | 0.92 | 0.44 | 0.00 | 0 | 1 | 0 | 3 |
| Ryan Gravenberch | MID | 77 | 0.50 | 0.47 | 0.00 | 1.12 | 0.52 | 0.00 | 0 | 0 | 0 | 2 |
| Ethan Ampadu | MID | 90 | 0.12 | 0.33 | 0.00 | 0.58 | 0.34 | 0.00 | 0 | 1 | 0 | 7 |
| Erling Haaland | FWD | 79 | 2.40 | 0.00 | 0.00 | 2.52 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Igor Thiago Nascimento Rodrigues | FWD | 90 | 2.20 | 0.00 | 0.00 | 1.69 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Hugo Ekitiké | FWD | 90 | 2.90 | 0.00 | 0.00 | 2.09 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |

### GW23 — pick_xi_score_xp_v2 · Σ total_pts=35

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Guglielmo Vicario | GKP | 90 | 0.00 | 1.80 | -0.25 | 0.00 | 1.38 | -0.31 | 0 | 0 | -1 | 1 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.45 | 2.00 | -0.25 | 0.40 | 1.56 | -0.26 | 0 | 0 | -1 | 1 |
| Cristian Romero | DEF | 90 | 0.45 | 1.80 | -0.25 | 0.35 | 1.38 | -0.31 | 6 | 0 | -1 | 11 |
| Trevoh Chalobah | DEF | 90 | 0.30 | 1.40 | -0.42 | 0.49 | 1.20 | -0.37 | 0 | 0 | 0 | 4 |
| Declan Rice | MID | 90 | 0.62 | 0.50 | 0.00 | 0.98 | 0.39 | 0.00 | 0 | 0 | 0 | 1 |
| Ryan Gravenberch | MID | 90 | 0.12 | 0.28 | 0.00 | 0.81 | 0.27 | 0.00 | 0 | 0 | 0 | 1 |
| Ethan Ampadu | MID | 90 | 0.00 | 0.25 | 0.00 | 0.47 | 0.30 | 0.00 | 0 | 0 | 0 | 4 |
| Antoine Semenyo | MID | 90 | 0.75 | 0.50 | 0.00 | 0.80 | 0.16 | 0.00 | 5 | 1 | 0 | 8 |
| Erling Haaland | FWD | 17 | 2.60 | 0.00 | 0.00 | 3.71 | 0.00 | 0.00 | 0 | 0 | 0 | 1 |
| Igor Thiago Nascimento Rodrigues | FWD | 90 | 2.10 | 0.00 | 0.00 | 2.43 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Hugo Ekitiké | FWD | 31 | 3.10 | 0.00 | 0.00 | 1.56 | 0.00 | 0.00 | 0 | 0 | 0 | 1 |

### GW24 — pick_xi_score_xp_v2 · Σ total_pts=56

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 1.80 | -0.23 | 0.00 | 1.81 | -0.20 | 0 | 4 | 0 | 6 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.15 | 1.80 | -0.23 | 0.37 | 1.81 | -0.20 | 0 | 4 | 0 | 8 |
| Trevoh Chalobah | DEF | 90 | 0.15 | 0.80 | -0.42 | 0.64 | 1.40 | -0.31 | 0 | 0 | -1 | 3 |
| Diogo Dalot Teixeira | DEF | 85 | 0.00 | 1.80 | -0.28 | 0.23 | 1.45 | -0.29 | 0 | 0 | -1 | 2 |
| Marc Guéhi | DEF | 90 | 0.15 | 1.30 | -0.38 | 0.26 | 0.87 | -0.53 | 0 | 0 | -1 | 3 |
| Declan Rice | MID | 90 | 0.25 | 0.45 | 0.00 | 0.91 | 0.45 | 0.00 | 0 | 1 | 0 | 3 |
| Ryan Gravenberch | MID | 90 | 0.38 | 0.28 | 0.00 | 0.81 | 0.30 | 0.00 | 0 | 0 | 0 | 4 |
| Antoine Semenyo | MID | 90 | 1.00 | 0.33 | 0.00 | 1.00 | 0.26 | 0.00 | 5 | 0 | 0 | 7 |
| Erling Haaland | FWD | 90 | 2.90 | 0.00 | 0.00 | 2.50 | 0.00 | 0.00 | 0 | 0 | 0 | 5 |
| Igor Thiago Nascimento Rodrigues | FWD | 90 | 1.70 | 0.00 | 0.00 | 1.82 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Hugo Ekitiké | FWD | 83 | 2.00 | 0.00 | 0.00 | 1.55 | 0.00 | 0.00 | 8 | 0 | 0 | 13 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp --n 40 --goal-weight rate
```

