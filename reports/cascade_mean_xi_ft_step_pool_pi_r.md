# Stepped ≥60′ selection pool — xP(G, CS, GA) vs actual

Season **2025-26** · horizon 8–19 · commit steps [8, 9, 10, 11, 12] · N=40 · seed=7 · goal_weight=`rate` (π∝r (minutes gate only)) · wall 416.0s

## Protocol

- At each decision GW t: rebuild **selection** pool = players with as-of mean minutes > 60 (and ≥2 apps) before t.
- **Share** pool = full club with m>0 (impact subs on generative dice).
- **Goal shares**: `rate` — π∝r (minutes gate only). Assists stay ∝rA·m/90.
- That as-of state (rates, minutes, both pools) is **projected forward** on strands for t…horizon end until the next step refreshes it.
- Mean of N generative boards → `xi_ft` plan; commit week-t XI.
- Residual = xP − actual on that XI. **GA** = goals-against points (`xp_gc` / FPL GC).

- **Seed owned**: `data/processed/l3_l4_gw08_seed_2025_26.json` (hold GW8: 0 xfers; seed XI+captain if present else `pick_xi`; `advance_ft(ft,0)`). Source: solve_week_sigma score_xp_v2 (L3/L4 WC open).

![per-match resid](/opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid_seeded_pi_r.png)

## Overall (committed GW8–12 XI)

- Player-GWs: **55**

| component | Σ xP | Σ actual | Σ resid | mean / match | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 51.4 | 57.0 | -5.5 | -0.101 | 1.462 |
| CS | 46.9 | 58.0 | -11.1 | -0.201 | 1.013 |
| GA | -6.3 | -7.0 | +0.7 | +0.012 | 0.180 |

## Per committed GW

| GW | select n | share n | xfers | hits | r_G | r_CS | r_GA | ΣxP_G | ΣP_G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 162 | 379 | 0 | 0 | -1.22 | -1.64 | -0.12 | 6.6 | 20.0 |
| 9 | 163 | 377 | 5 | 12 | +0.11 | -0.71 | -0.15 | 11.2 | 10.0 |
| 10 | 154 | 381 | 1 | 0 | +0.20 | -0.18 | -0.11 | 10.2 | 8.0 |
| 11 | 160 | 379 | 0 | 0 | -0.46 | +0.71 | +0.42 | 9.9 | 15.0 |
| 12 | 159 | 371 | 2 | 0 | +0.88 | +0.82 | +0.03 | 13.6 | 4.0 |

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

### GW9 — buy ['Bryan Mbeumo', 'Bruno Borges Fernandes', 'Gabriel dos Santos Magalhães', 'Marcos Senesi Barón', 'Antoine Semenyo'] / sell ['Jaidon Anthony', 'Trevoh Chalobah', 'Matheus Nunes', 'Chris Rigg', 'Jhon Arias'] · ft 2→1 · hits=12

**XI**
1. David Raya Martín
2. Gabriel dos Santos Magalhães
3. Jurriën Timber
4. Marcos Senesi Barón
5. Antoine Semenyo
6. Bryan Mbeumo
7. Ismaïla Sarr
8. Erling Haaland
9. Bruno Borges Fernandes
10. Omar Alderete
11. Nick Woltemade

**Bench**
1. Ethan Ampadu
2. Guglielmo Vicario
3. Junior Kroupi
4. Daniel Ballard

### GW10 — buy ['Dominik Szoboszlai'] / sell ['Antoine Semenyo'] · ft 1→1 · hits=0

**XI**
1. David Raya Martín
2. Gabriel dos Santos Magalhães
3. Jurriën Timber
4. Bryan Mbeumo
5. Ismaïla Sarr
6. Dominik Szoboszlai
7. Erling Haaland
8. Bruno Borges Fernandes
9. Omar Alderete
10. Nick Woltemade
11. Daniel Ballard

**Bench**
1. Marcos Senesi Barón
2. Ethan Ampadu
3. Guglielmo Vicario
4. Junior Kroupi

### GW11 — buy [] / sell [] · ft 1→2 · hits=0

**XI**
1. David Raya Martín
2. Gabriel dos Santos Magalhães
3. Jurriën Timber
4. Marcos Senesi Barón
5. Bryan Mbeumo
6. Ismaïla Sarr
7. Dominik Szoboszlai
8. Erling Haaland
9. Bruno Borges Fernandes
10. Junior Kroupi
11. Daniel Ballard

**Bench**
1. Ethan Ampadu
2. Guglielmo Vicario
3. Omar Alderete
4. Nick Woltemade

### GW12 — buy ['Igor Thiago Nascimento Rodrigues', "Nico O'Reilly"] / sell ['Omar Alderete', 'Nick Woltemade'] · ft 2→1 · hits=0

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
1. Ethan Ampadu
2. Nico O'Reilly
3. Guglielmo Vicario
4. Daniel Ballard


## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_step_pool_resid --n 40 --steps 8,9,10,11,12 --goal-weight rate --seed-owned data/processed/l3_l4_gw08_seed_2025_26.json --out /opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_pi_r.json
```

