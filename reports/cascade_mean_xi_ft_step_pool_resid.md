# Stepped ≥60′ selection pool — xP(G, CS, GA) vs actual

Season **2025-26** · horizon 8–19 · commit steps [8, 9, 10, 11, 12] · N=40 · seed=7 · wall 496.6s

## Protocol

- At each decision GW t: rebuild **selection** pool = players with as-of mean minutes > 60 (and ≥2 apps) before t.
- **Share** pool = full club with m>0 (impact subs on generative dice).
- That as-of state (rates, minutes, both pools) is **projected forward** on strands for t…horizon end until the next step refreshes it.
- Mean of N generative boards → `xi_ft` plan; commit week-t XI.
- Residual = xP − actual on that XI. **GA** = goals-against points (`xp_gc` / FPL GC).

- **Seed owned**: `data/processed/l3_l4_gw08_seed_2025_26.json` (hold GW8: 0 xfers; seed XI+captain if present else `pick_xi`; `advance_ft(ft,0)`). Source: solve_week_sigma score_xp_v2 (L3/L4 WC open).

![per-match resid](/opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid_seeded.png)

## Overall (committed GW8–12 XI)

- Player-GWs: **55**

| component | Σ xP | Σ actual | Σ resid | mean / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 72.5 | 61.0 | +11.5 | +0.208 | 1.605 |
| CS | 47.6 | 62.0 | -14.4 | -0.261 | 1.062 |
| GA | -4.7 | -5.0 | +0.3 | +0.005 | 0.147 |

## Per committed GW

| GW | select n | share n | xfers | hits | r_G | r_CS | r_GA | ΣxP_G | ΣP_G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 162 | 379 | 0 | 0 | -1.05 | -1.64 | -0.12 | 8.5 | 20.0 |
| 9 | 163 | 377 | 6 | 16 | +0.49 | -0.64 | +0.00 | 15.4 | 10.0 |
| 10 | 154 | 381 | 1 | 0 | +0.65 | -0.17 | -0.11 | 15.2 | 8.0 |
| 11 | 160 | 379 | 1 | 0 | -0.17 | +0.33 | +0.22 | 17.2 | 19.0 |
| 12 | 159 | 371 | 1 | 0 | +1.11 | +0.82 | +0.03 | 16.2 | 4.0 |

## Squads (XI + bench)

### GW8 — buy [] / sell [] · ft 1→2 · hits=0

**XI**
1. David Raya Martín
2. Jurriën Timber
3. Junior Kroupi
4. Jaidon Anthony
5. Trevoh Chalobah
6. Ismaïla Sarr
7. Matheus Nunes
8. Erling Haaland
9. Daniel Ballard
10. Omar Alderete
11. Nick Woltemade

**Bench**
1. Ethan Ampadu
2. Chris Rigg
3. Guglielmo Vicario
4. Jhon Arias

### GW9 — buy ['Bryan Mbeumo', 'Michael Keane', 'Bruno Borges Fernandes', 'Gabriel dos Santos Magalhães', 'Marcos Senesi Barón', 'Antoine Semenyo'] / sell ['Jaidon Anthony', 'Trevoh Chalobah', 'Ethan Ampadu', 'Matheus Nunes', 'Daniel Ballard', 'Jhon Arias'] · ft 2→1 · hits=16

**XI**
1. David Raya Martín
2. Gabriel dos Santos Magalhães
3. Jurriën Timber
4. Marcos Senesi Barón
5. Antoine Semenyo
6. Bryan Mbeumo
7. Ismaïla Sarr
8. Michael Keane
9. Erling Haaland
10. Bruno Borges Fernandes
11. Nick Woltemade

**Bench**
1. Guglielmo Vicario
2. Omar Alderete
3. Junior Kroupi
4. Chris Rigg

### GW10 — buy ['Dominik Szoboszlai'] / sell ['Antoine Semenyo'] · ft 1→1 · hits=0

**XI**
1. David Raya Martín
2. Gabriel dos Santos Magalhães
3. Jurriën Timber
4. Bryan Mbeumo
5. Ismaïla Sarr
6. Michael Keane
7. Dominik Szoboszlai
8. Erling Haaland
9. Bruno Borges Fernandes
10. Omar Alderete
11. Nick Woltemade

**Bench**
1. Marcos Senesi Barón
2. Guglielmo Vicario
3. Junior Kroupi
4. Chris Rigg

### GW11 — buy ['João Pedro Junqueira de Jesus'] / sell ['Nick Woltemade'] · ft 1→1 · hits=0

**XI**
1. David Raya Martín
2. Gabriel dos Santos Magalhães
3. Jurriën Timber
4. Bryan Mbeumo
5. João Pedro Junqueira de Jesus
6. Ismaïla Sarr
7. Michael Keane
8. Dominik Szoboszlai
9. Erling Haaland
10. Bruno Borges Fernandes
11. Junior Kroupi

**Bench**
1. Marcos Senesi Barón
2. Guglielmo Vicario
3. Omar Alderete
4. Chris Rigg

### GW12 — buy ['Igor Thiago Nascimento Rodrigues'] / sell ['João Pedro Junqueira de Jesus'] · ft 1→1 · hits=0

**XI**
1. David Raya Martín
2. Gabriel dos Santos Magalhães
3. Jurriën Timber
4. Marcos Senesi Barón
5. Bryan Mbeumo
6. Igor Thiago Nascimento Rodrigues
7. Ismaïla Sarr
8. Dominik Szoboszlai
9. Erling Haaland
10. Bruno Borges Fernandes
11. Junior Kroupi

**Bench**
1. Michael Keane
2. Guglielmo Vicario
3. Omar Alderete
4. Chris Rigg


## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_step_pool_resid --n 40 --steps 8,9,10,11,12 --seed-owned data/processed/l3_l4_gw08_seed_2025_26.json --out /opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid_seeded.json
```

