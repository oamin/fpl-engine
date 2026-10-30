# Cascade π∝r vs L3/L4 — player xP(G, CS, GA) + actual (no captain)

Season **2025-26** · GW8–13 (4 weeks) · N=8 · goal_weight=`rate` (π∝r (minutes gate only)) · wall 19.6s

## What this is

- Shared L3 path XI from WC open (`l3_l4_gw08_seed_2025_26.json`); later weeks = L3 buys/sells + `pick_xi` on `score_xp_v2`.
- Both scorers on **that same XI** (no captain, no hits):
  - **cascade** — generative mean `xp_goals` / `xp_cs` / `xp_gc` (goal shares `rate`)
  - **l3_v2** — as-of `score_xp_v2` centreline components
  - **actual** — FPL `P_G` / `P_CS` / `P_GA`
- Cascade rolls **decision GW only** (season map).

![resid](/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_resid_gw8-13.png)

## Block totals (Σ over XI)

| block | n | casc G | l3 G | P_G | casc−P G | l3−P G | casc CS | l3 CS | P_CS | casc−P CS | l3−P CS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GW8-12 | 3 | 18.1 | 23.7 | 36.0 | -17.9 | -12.3 | 26.9 | 23.3 | 54.0 | -27.1 | -30.7 |
| GW13-19 | 1 | 9.9 | 7.4 | 6.0 | +3.9 | +1.4 | 8.8 | 5.9 | 1.0 | +7.8 | +4.9 |

**Overall** (44 player-GWs): casc ΣG/CS/GA = 28.0/35.6/-3.9; l3 = 31.1/29.1/-4.7; actual = 42.0/55.0/-4.0.

- casc − actual G: **-14.0**
- l3 − actual G: **-10.9**
- casc − actual CS: **-19.4**
- l3 − actual CS: **-25.9**

| component | MAE casc | MAE l3_v2 |
| --- | ---: | ---: |
| G | 1.364 | 1.241 |
| CS | 0.912 | 1.003 |
| GA | 0.122 | 0.157 |

## Per GW (Σ on XI)

| GW | casc G | l3 G | P_G | casc−P G | l3−P G | casc CS | l3 CS | P_CS | casc−P CS | l3−P CS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4.6 | 7.3 | 20.0 | -15.4 | -12.7 | 8.2 | 6.8 | 25.0 | -16.8 | -18.2 |
| 9 | 6.1 | 8.1 | 4.0 | +2.1 | +4.1 | 8.5 | 8.3 | 12.0 | -3.5 | -3.7 |
| 10 | 7.4 | 8.3 | 12.0 | -4.6 | -3.7 | 10.1 | 8.2 | 17.0 | -6.9 | -8.8 |
| 13 | 9.9 | 7.4 | 6.0 | +3.9 | +1.4 | 8.8 | 5.9 | 1.0 | +7.8 | +4.9 |

## Player rows

### GW8 — seed_l3_l4_wc · Σ total_pts=78

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.50 | -0.12 | 0.00 | 1.79 | -0.20 | 0 | 4 | 0 | 6 |
| Jurriën Timber | DEF | 90 | 0.00 | 2.50 | -0.12 | 0.35 | 1.79 | -0.20 | 0 | 4 | 0 | 6 |
| Junior Kroupi | FWD | 74 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 | 0.00 | 8 | 0 | 0 | 12 |
| Jaidon Anthony | MID | 82 | 0.62 | 0.25 | 0.00 | 0.84 | 0.26 | 0.00 | 0 | 1 | 0 | 2 |
| Trevoh Chalobah | DEF | 90 | 0.75 | 0.00 | -0.25 | 0.44 | 1.19 | -0.38 | 0 | 4 | 0 | 6 |
| Ismaïla Sarr | MID | 90 | 1.25 | 0.00 | 0.00 | 0.77 | 0.31 | 0.00 | 0 | 0 | 0 | 2 |
| Matheus Nunes | DEF | 90 | 0.00 | 2.00 | -0.12 | 0.31 | 0.00 | -0.18 | 0 | 4 | 0 | 7 |
| Erling Haaland | FWD | 90 | 1.50 | 0.00 | 0.00 | 2.70 | 0.00 | 0.00 | 8 | 0 | 0 | 13 |
| Daniel Ballard | DEF | 90 | 0.00 | 0.00 | -0.12 | 0.18 | 0.00 | -0.30 | 0 | 4 | 0 | 10 |
| Omar Alderete | DEF | 90 | 0.00 | 1.00 | -0.12 | 0.33 | 1.43 | -0.30 | 0 | 4 | 0 | 6 |
| Nick Woltemade | FWD | 90 | 0.50 | 0.00 | 0.00 | 1.33 | 0.00 | 0.00 | 4 | 0 | 0 | 8 |

### GW9 — pick_xi_score_xp_v2 · Σ total_pts=42

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.50 | -0.12 | 0.00 | 1.87 | -0.18 | 0 | 4 | 0 | 6 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.00 | 2.50 | -0.12 | 0.55 | 1.87 | -0.18 | 0 | 4 | 0 | 9 |
| Trevoh Chalobah | DEF | 90 | 0.00 | 0.00 | -0.25 | 0.63 | 1.91 | -0.18 | 0 | 0 | -1 | 1 |
| Jurriën Timber | DEF | 90 | 0.00 | 2.50 | -0.12 | 0.40 | 1.87 | -0.18 | 0 | 4 | 0 | 6 |
| Ethan Ampadu | MID | 90 | 0.00 | 0.38 | 0.00 | 0.52 | 0.40 | 0.00 | 0 | 0 | 0 | 4 |
| Jaidon Anthony | MID | 76 | 1.25 | 0.12 | 0.00 | 0.72 | 0.24 | 0.00 | 0 | 0 | 0 | 2 |
| Ismaïla Sarr | MID | 90 | 1.25 | 0.12 | 0.00 | 0.43 | 0.12 | 0.00 | 0 | 0 | 0 | 2 |
| Jhon Arias | MID | 67 | 0.62 | 0.38 | 0.00 | 0.53 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Erling Haaland | FWD | 90 | 3.00 | 0.00 | 0.00 | 2.22 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Nick Woltemade | FWD | 85 | 0.00 | 0.00 | 0.00 | 1.83 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Junior Kroupi | FWD | 68 | 0.00 | 0.00 | 0.00 | 0.29 | 0.00 | 0.00 | 4 | 0 | 0 | 6 |

### GW10 — pick_xi_score_xp_v2 · Σ total_pts=59

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.50 | -0.12 | 0.00 | 2.21 | -0.12 | 0 | 4 | 0 | 6 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.75 | 2.50 | -0.12 | 0.61 | 2.21 | -0.12 | 0 | 4 | 0 | 12 |
| Jurriën Timber | DEF | 90 | 0.00 | 2.50 | -0.12 | 0.45 | 2.21 | -0.12 | 0 | 4 | 0 | 6 |
| Trevoh Chalobah | DEF | 90 | 0.75 | 1.50 | -0.38 | 0.45 | 0.97 | -0.48 | 0 | 4 | 0 | 7 |
| Ismaïla Sarr | MID | 90 | 1.88 | 0.50 | 0.00 | 1.01 | 0.31 | 0.00 | 0 | 1 | 0 | 3 |
| Ethan Ampadu | MID | 90 | 0.00 | 0.12 | 0.00 | 0.35 | 0.18 | 0.00 | 0 | 0 | 0 | 1 |
| Jaidon Anthony | MID | 61 | 0.00 | 0.00 | 0.00 | 0.43 | 0.10 | 0.00 | 0 | 0 | 0 | 2 |
| Jhon Arias | MID | 45 | 0.00 | 0.50 | 0.00 | 0.32 | 0.00 | 0.00 | 0 | 0 | 0 | 1 |
| Erling Haaland | FWD | 81 | 2.00 | 0.00 | 0.00 | 2.88 | 0.00 | 0.00 | 8 | 0 | 0 | 13 |
| Danny Welbeck | FWD | 81 | 1.00 | 0.00 | 0.00 | 1.51 | 0.00 | 0.00 | 4 | 0 | 0 | 6 |
| Junior Kroupi | FWD | 61 | 1.00 | 0.00 | 0.00 | 0.28 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |

### GW13 — pick_xi_score_xp_v2 · Σ total_pts=29

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Guglielmo Vicario | GKP | 90 | 0.00 | 3.00 | 0.00 | 0.00 | 1.89 | -0.18 | 0 | 0 | -1 | 1 |
| Marc Guéhi | DEF | 90 | 0.00 | 1.00 | -0.75 | 0.46 | 1.64 | -0.24 | 0 | 0 | -1 | 0 |
| Trevoh Chalobah | DEF | 90 | 0.00 | 1.50 | -0.25 | 0.47 | 1.38 | -0.31 | 6 | 0 | 0 | 9 |
| Jurriën Timber | DEF | 90 | 0.00 | 0.50 | -0.38 | 0.22 | 0.84 | -0.54 | 0 | 0 | 0 | 2 |
| Daniel Ballard | DEF | 90 | 0.75 | 2.50 | -0.25 | 0.40 | 0.00 | -0.28 | 0 | 0 | -1 | 3 |
| Ethan Ampadu | MID | 90 | 0.00 | 0.12 | 0.00 | 0.07 | 0.05 | 0.00 | 0 | 0 | 0 | 2 |
| Jaidon Anthony | MID | 15 | 0.00 | 0.12 | 0.00 | 0.22 | 0.08 | 0.00 | 0 | 0 | 0 | 4 |
| Donyell Malen | MID | 74 | 3.12 | 0.00 | 0.00 | 0.62 | 0.00 | 0.00 | 0 | 1 | 0 | 3 |
| Erling Haaland | FWD | 90 | 4.00 | 0.00 | 0.00 | 3.82 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Danny Welbeck | FWD | 64 | 0.50 | 0.00 | 0.00 | 0.80 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Junior Kroupi | FWD | 10 | 1.50 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0 | 0 | 0 | 1 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp --n 8 --goal-weight rate --steps 8,9,10,13 --decision-gw-only
```

