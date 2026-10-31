# Cascade π∝r vs L3/L4 — player xP(G, CS, GA) + actual (no captain)

Season **2025-26** · GW30–34 (3 weeks) · N=8 · goal_weight=`rate` (π∝r (minutes gate only)) · wall 19.3s

## What this is

- Shared L3 path XI from WC open (`l3_l4_gw08_seed_2025_26.json`); later weeks = L3 buys/sells + `pick_xi` on `score_xp_v2`.
- Both scorers on **that same XI** (no captain, no hits):
  - **cascade** — generative mean `xp_goals` / `xp_cs` / `xp_gc` (goal shares `rate`)
  - **l3_v2** — as-of `score_xp_v2` centreline components
  - **actual** — FPL `P_G` / `P_CS` / `P_GA`
- Cascade rolls **decision GW only** (season map).

![resid](/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_resid_gw30-34.png)

## Block totals (Σ over XI)

| block | n | casc G | l3 G | P_G | casc−P G | l3−P G | casc CS | l3 CS | P_CS | casc−P CS | l3−P CS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GW27-33 | 2 | 14.5 | 20.0 | 14.0 | +0.5 | +6.0 | 13.9 | 14.9 | 14.0 | -0.1 | +0.9 |
| GW34-38 | 1 | 9.5 | 9.0 | 14.0 | -4.5 | -5.0 | 4.2 | 6.2 | 6.0 | -1.8 | +0.2 |

**Overall** (33 player-GWs): casc ΣG/CS/GA = 24.0/18.1/-4.5; l3 = 29.0/21.1/-4.8; actual = 28.0/20.0/-3.0.

- casc − actual G: **-4.0**
- l3 − actual G: **+1.0**
- casc − actual CS: **-1.9**
- l3 − actual CS: **+1.1**

| component | MAE casc | MAE l3_v2 |
| --- | ---: | ---: |
| G | 1.121 | 1.243 |
| CS | 0.670 | 0.795 |
| GA | 0.159 | 0.168 |

## Per GW (Σ on XI)

| GW | casc G | l3 G | P_G | casc−P G | l3−P G | casc CS | l3 CS | P_CS | casc−P CS | l3−P CS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 9.8 | 12.3 | 4.0 | +5.8 | +8.3 | 9.5 | 8.6 | 13.0 | -3.5 | -4.4 |
| 31 | 4.8 | 7.7 | 10.0 | -5.2 | -2.3 | 4.4 | 6.3 | 1.0 | +3.4 | +5.3 |
| 34 | 9.5 | 9.0 | 14.0 | -4.5 | -5.0 | 4.2 | 6.2 | 6.0 | -1.8 | +0.2 |

## Player rows

### GW30 — pick_xi_score_xp_v2 · Σ total_pts=54

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| David Raya Martín | GKP | 90 | 0.00 | 2.00 | -0.12 | 0.00 | 2.12 | -0.14 | 0 | 4 | 0 | 7 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.75 | 2.00 | -0.12 | 0.43 | 2.12 | -0.14 | 0 | 4 | 0 | 9 |
| Marc Guéhi | DEF | 90 | 0.00 | 1.50 | -0.50 | 0.38 | 1.25 | -0.36 | 0 | 0 | 0 | 2 |
| Diogo Dalot Teixeira | DEF | 90 | 0.00 | 1.50 | -0.50 | 0.21 | 1.31 | -0.33 | 0 | 0 | 0 | 2 |
| Malick Thiaw | DEF | 90 | 0.00 | 1.50 | -0.38 | 0.33 | 0.53 | -0.76 | 0 | 4 | 0 | 9 |
| Declan Rice | MID | 90 | 0.62 | 0.50 | 0.00 | 0.91 | 0.53 | 0.00 | 0 | 1 | 0 | 3 |
| Bruno Borges Fernandes | MID | 90 | 1.25 | 0.38 | 0.00 | 1.11 | 0.33 | 0.00 | 0 | 0 | 0 | 10 |
| Ryan Gravenberch | MID | 90 | 0.62 | 0.12 | 0.00 | 1.03 | 0.41 | 0.00 | 0 | 0 | 0 | 2 |
| Erling Haaland | FWD | 90 | 3.50 | 0.00 | 0.00 | 2.60 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| João Pedro Junqueira de Jesus | FWD | 90 | 0.00 | 0.00 | 0.00 | 2.28 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Igor Thiago Nascimento Rodrigues | FWD | 90 | 3.00 | 0.00 | 0.00 | 3.04 | 0.00 | 0.00 | 4 | 0 | 0 | 6 |

### GW31 — pick_xi_score_xp_v2 · Σ total_pts=42

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Guglielmo Vicario | GKP | 90 | 0.00 | 2.50 | -0.38 | 0.00 | 1.14 | -0.40 | 0 | 0 | -1 | 2 |
| Malick Thiaw | DEF | 36 | 1.50 | 0.00 | -0.25 | 0.45 | 1.60 | -0.25 | 0 | 0 | -1 | 0 |
| Calvin Bassey | DEF | 90 | 0.00 | 1.50 | -0.38 | 0.27 | 1.67 | -0.23 | 0 | 0 | 0 | 2 |
| Diogo Dalot Teixeira | DEF | 90 | 0.00 | 0.00 | -0.50 | 0.20 | 0.95 | -0.48 | 0 | 0 | -1 | 1 |
| Bruno Borges Fernandes | MID | 90 | 0.62 | 0.00 | 0.00 | 1.06 | 0.24 | 0.00 | 5 | 0 | 0 | 13 |
| Harry Wilson | MID | 86 | 0.62 | 0.00 | 0.00 | 1.12 | 0.42 | 0.00 | 5 | 0 | 0 | 9 |
| Ryan Gravenberch | MID | 90 | 0.00 | 0.38 | 0.00 | 0.66 | 0.26 | 0.00 | 0 | 0 | 0 | 3 |
| Ross Barkley | MID | 66 | 0.00 | 0.00 | 0.00 | 0.16 | 0.00 | 0.00 | 0 | 1 | 0 | 3 |
| Chris Rigg | MID | 90 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 | 0 | 0 | 0 | 3 |
| João Pedro Junqueira de Jesus | FWD | 77 | 1.00 | 0.00 | 0.00 | 1.84 | 0.00 | 0.00 | 0 | 0 | 0 | 2 |
| Igor Thiago Nascimento Rodrigues | FWD | 90 | 1.00 | 0.00 | 0.00 | 1.90 | 0.00 | 0.00 | 0 | 0 | 0 | 4 |

### GW34 — pick_xi_score_xp_v2 · Σ total_pts=52

| player | pos | m | casc G | casc CS | casc GA | l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Guglielmo Vicario | GKP | 0 | 0.00 | 0.00 | -0.25 | 0.00 | 1.35 | -0.32 | 0 | 0 | 0 | 0 |
| Gabriel dos Santos Magalhães | DEF | 90 | 0.00 | 1.50 | 0.00 | 0.46 | 1.53 | -0.27 | 0 | 4 | 0 | 8 |
| Virgil van Dijk | DEF | 90 | 0.75 | 1.50 | -0.12 | 0.58 | 1.49 | -0.28 | 0 | 0 | 0 | 4 |
| Malick Thiaw | DEF | 90 | 0.00 | 0.00 | -1.00 | 0.24 | 0.46 | -0.83 | 0 | 0 | 0 | 2 |
| Declan Rice | MID | 90 | 0.62 | 0.38 | 0.00 | 1.06 | 0.38 | 0.00 | 0 | 1 | 0 | 2 |
| Bruno Borges Fernandes | MID | 90 | 0.62 | 0.12 | 0.00 | 1.13 | 0.28 | 0.00 | 0 | 0 | 0 | 5 |
| Carlos Henrique Casimiro | MID | 90 | 0.62 | 0.12 | 0.00 | 0.88 | 0.28 | 0.00 | 5 | 0 | 0 | 12 |
| Kiernan Dewsbury-Hall | MID | 90 | 0.62 | 0.50 | 0.00 | 0.66 | 0.24 | 0.00 | 5 | 0 | 0 | 8 |
| Harry Wilson | MID | 90 | 1.25 | 0.12 | 0.00 | 0.81 | 0.22 | 0.00 | 0 | 1 | 0 | 2 |
| Igor Thiago Nascimento Rodrigues | FWD | 90 | 3.00 | 0.00 | 0.00 | 1.93 | 0.00 | 0.00 | 0 | 0 | 0 | 1 |
| Benjamin Sesko | FWD | 87 | 2.00 | 0.00 | 0.00 | 1.23 | 0.00 | 0.00 | 4 | 0 | 0 | 8 |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp --n 8 --goal-weight rate --steps 30,31,34 --decision-gw-only
```

