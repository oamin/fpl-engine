# Cascade vs L3/L4 — player xP(G, CS, GA) + actual (no captain)

Season **2025-26** · GW [8, 9, 10, 11, 12] · N=40 · wall 431.5s

## What this is

- **Same GW8 team** for both arms: L3/L4 wildcard open (`l3_l4_gw08_seed_2025_26.json`).
- GW9–12 follow the locked **L3** transfer path; XI = `pick_xi` on `score_xp_v2` among that owned 15.
- Both scorers are read on **that same XI** (no captain, no hits):
  - **cascade** — generative mean `xp_goals` / `xp_cs` / `xp_gc`
  - **l3_v2** — as-of `score_xp_v2` centreline components (L4 μ_post≡v2 when mode=none)
  - **actual** — FPL `P_G` / `P_CS` / `P_GA` and full `total_points`

## XI totals by GW (no captain)

| GW | casc ΣG | casc ΣCS | casc ΣGA | l3 ΣG | l3 ΣCS | l3 ΣGA | P ΣG | P ΣCS | P ΣGA | Σ total_pts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8.5 | 7.0 | -1.4 | 7.3 | 6.8 | -1.6 | 20.0 | 25.0 | 0.0 | 78 |
| 9 | 10.5 | 7.1 | -0.6 | 8.1 | 8.3 | -0.7 | 4.0 | 12.0 | -1.0 | 42 |
| 10 | 11.4 | 10.2 | -0.9 | 8.3 | 8.2 | -0.8 | 12.0 | 17.0 | 0.0 | 59 |
| 11 | 12.8 | 9.3 | -0.4 | 8.0 | 9.1 | -0.6 | 4.0 | 5.0 | -3.0 | 23 |
| 12 | 10.6 | 8.8 | -0.5 | 8.0 | 7.9 | -0.8 | 4.0 | 9.0 | 0.0 | 40 |

**Overall** (55 player-GWs): casc ΣG/CS/GA = 53.7/42.4/-3.9; l3 = 39.7/40.3/-4.5; actual = 44.0/68.0/-4.0; Σ total_points = 242.

| component | MAE casc | MAE l3_v2 |
| --- | ---: | ---: |
| G | 1.246 | 1.052 |
| CS | 0.980 | 1.034 |
| GA | 0.126 | 0.135 |

## Player rows

### GW8 — seed_l3_l4_wc · Σ total_pts=78

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 1.50 | -0.17 | 0.00 | 1.79 | -0.20 | 0 | 4 | 0 | 6 |
| Jurriën Timber | DEF | 90 | 0.15 | 1.50 | -0.17 | 0.35 | 1.79 | -0.20 | 0 | 4 | 0 | 6 |
| Junior Kroupi | FWD | 74 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 | 0.00 | 8 | 0 | 0 | 12 |
| Jaidon Anthony | MID | 82 | 0.88 | 0.28 | 0.00 | 0.84 | 0.26 | 0.00 | 0 | 1 | 0 | 2 |
| Trevoh Chalobah | DEF | 90 | 1.05 | 0.00 | -0.30 | 0.44 | 1.19 | -0.38 | 0 | 4 | 0 | 6 |
| Ismaïla Sarr | MID | 90 | 0.75 | 0.00 | 0.00 | 0.77 | 0.31 | 0.00 | 0 | 0 | 0 | 2 |
| Matheus Nunes | DEF | 90 | 0.75 | 2.20 | -0.17 | 0.31 | 0.00 | -0.18 | 0 | 4 | 0 | 7 |
| Erling Haaland | FWD | 90 | 2.80 | 0.00 | 0.00 | 2.70 | 0.00 | 0.00 | 8 | 0 | 0 | 13 |
| Daniel Ballard | DEF | 90 | 0.00 | 0.00 | -0.28 | 0.18 | 0.00 | -0.30 | 0 | 4 | 0 | 10 |
| Omar Alderete | DEF | 90 | 0.30 | 1.50 | -0.28 | 0.33 | 1.43 | -0.30 | 0 | 4 | 0 | 6 |
| Nick Woltemade | FWD | 90 | 1.80 | 0.00 | 0.00 | 1.33 | 0.00 | 0.00 | 4 | 0 | 0 | 8 |

### GW9 — pick_xi_score_xp_v2 · Σ total_pts=42

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.00 | -0.12 | 0.00 | 1.87 | -0.18 | 0 | 4 | 0 | 6 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.75 | 2.00 | -0.12 | 0.55 | 1.87 | -0.18 | 0 | 4 | 0 | 9 |
| Trevoh Chalobah | DEF | 90 | 0.60 | 0.00 | -0.23 | 0.63 | 1.91 | -0.18 | 0 | 0 | -1 | 1 |
| Jurriën Timber | DEF | 90 | 0.30 | 2.00 | -0.12 | 0.40 | 1.87 | -0.18 | 0 | 4 | 0 | 6 |
| Ethan Ampadu | MID | 90 | 0.50 | 0.33 | 0.00 | 0.52 | 0.40 | 0.00 | 0 | 0 | 0 | 4 |
| Jaidon Anthony | MID | 76 | 1.25 | 0.17 | 0.00 | 0.72 | 0.24 | 0.00 | 0 | 0 | 0 | 2 |
| Ismaïla Sarr | MID | 90 | 1.25 | 0.15 | 0.00 | 0.43 | 0.12 | 0.00 | 0 | 0 | 0 | 2 |
| Jhon Arias | MID | 67 | 0.38 | 0.45 | 0.00 | 0.53 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Erling Haaland | FWD | 90 | 3.10 | 0.00 | 0.00 | 2.22 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Nick Woltemade | FWD | 85 | 2.40 | 0.00 | 0.00 | 1.83 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Junior Kroupi | FWD | 68 | 0.00 | 0.00 | 0.00 | 0.29 | 0.00 | 0.00 | 4 | 0 | 0 | 6 |

### GW10 — pick_xi_score_xp_v2 · Σ total_pts=59

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.70 | -0.15 | 0.00 | 2.21 | -0.12 | 0 | 4 | 0 | 6 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.75 | 2.70 | -0.15 | 0.61 | 2.21 | -0.12 | 0 | 4 | 0 | 12 |
| Jurriën Timber | DEF | 90 | 0.60 | 2.70 | -0.15 | 0.45 | 2.21 | -0.12 | 0 | 4 | 0 | 6 |
| Trevoh Chalobah | DEF | 90 | 0.75 | 1.10 | -0.47 | 0.45 | 0.97 | -0.48 | 0 | 4 | 0 | 7 |
| Ismaïla Sarr | MID | 90 | 1.88 | 0.33 | 0.00 | 1.01 | 0.31 | 0.00 | 0 | 1 | 0 | 3 |
| Ethan Ampadu | MID | 90 | 0.00 | 0.28 | 0.00 | 0.35 | 0.18 | 0.00 | 0 | 0 | 0 | 1 |
| Jaidon Anthony | MID | 61 | 0.50 | 0.12 | 0.00 | 0.43 | 0.10 | 0.00 | 0 | 0 | 0 | 2 |
| Jhon Arias | MID | 45 | 0.12 | 0.28 | 0.00 | 0.32 | 0.00 | 0.00 | 0 | 0 | 0 | 1 |
| Erling Haaland | FWD | 81 | 3.60 | 0.00 | 0.00 | 2.88 | 0.00 | 0.00 | 8 | 0 | 0 | 13 |
| Danny Welbeck | FWD | 81 | 1.80 | 0.00 | 0.00 | 1.51 | 0.00 | 0.00 | 4 | 0 | 0 | 6 |
| Junior Kroupi | FWD | 61 | 1.40 | 0.00 | 0.00 | 0.28 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |

### GW11 — pick_xi_score_xp_v2 · Σ total_pts=23

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.30 | -0.07 | 0.00 | 2.20 | -0.13 | 0 | 0 | -1 | 1 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.15 | 2.30 | -0.07 | 0.51 | 2.20 | -0.13 | 0 | 0 | -1 | 1 |
| Jurriën Timber | DEF | 90 | 0.75 | 2.30 | -0.07 | 0.38 | 2.20 | -0.13 | 0 | 0 | -1 | 1 |
| Trevoh Chalobah | DEF | 90 | 1.50 | 1.80 | -0.20 | 0.70 | 1.84 | -0.19 | 0 | 4 | 0 | 6 |
| Ismaïla Sarr | MID | 90 | 2.38 | 0.25 | 0.00 | 0.93 | 0.26 | 0.00 | 0 | 1 | 0 | 3 |
| Jaidon Anthony | MID | 80 | 0.88 | 0.10 | 0.00 | 0.72 | 0.21 | 0.00 | 0 | 0 | 0 | 2 |
| Ethan Ampadu | MID | 90 | 0.50 | 0.20 | 0.00 | 0.36 | 0.25 | 0.00 | 0 | 0 | 0 | 2 |
| Jhon Arias | MID | 0 | 0.00 | 0.07 | 0.00 | 0.28 | 0.00 | 0.00 | 0 | 0 | 0 | 0 |
| Erling Haaland | FWD | 90 | 2.70 | 0.00 | 0.00 | 2.51 | 0.00 | 0.00 | 4 | 0 | 0 | 4 |
| Danny Welbeck | FWD | 84 | 1.30 | 0.00 | 0.00 | 1.22 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Junior Kroupi | FWD | 27 | 2.60 | 0.00 | 0.00 | 0.35 | 0.00 | 0.00 | 0 | 0 | 0 | 1 |

### GW12 — pick_xi_score_xp_v2 · Σ total_pts=40

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.40 | -0.12 | 0.00 | 1.93 | -0.17 | 0 | 0 | 0 | 2 |
| Jurriën Timber | DEF | 90 | 0.75 | 2.40 | -0.12 | 0.42 | 1.93 | -0.17 | 0 | 0 | 0 | 5 |
| Trevoh Chalobah | DEF | 90 | 1.05 | 1.80 | -0.10 | 0.61 | 1.80 | -0.20 | 0 | 4 | 0 | 6 |
| Marc Guéhi | DEF | 90 | 0.30 | 1.50 | -0.17 | 0.37 | 1.44 | -0.29 | 0 | 4 | 0 | 5 |
| Ismaïla Sarr | MID | 90 | 1.38 | 0.38 | 0.00 | 0.97 | 0.36 | 0.00 | 0 | 1 | 0 | 3 |
| Ethan Ampadu | MID | 90 | 0.25 | 0.23 | 0.00 | 0.37 | 0.27 | 0.00 | 0 | 0 | 0 | 4 |
| Jaidon Anthony | MID | 71 | 0.38 | 0.07 | 0.00 | 0.56 | 0.13 | 0.00 | 0 | 0 | 0 | 2 |
| Jhon Arias | MID | 15 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0 | 0 | 0 | 1 |
| Erling Haaland | FWD | 90 | 1.30 | 0.00 | 0.00 | 2.15 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Danny Welbeck | FWD | 90 | 1.90 | 0.00 | 0.00 | 1.61 | 0.00 | 0.00 | 4 | 0 | 0 | 8 |
| Junior Kroupi | FWD | 85 | 3.30 | 0.00 | 0.00 | 0.57 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp --n 40
```

