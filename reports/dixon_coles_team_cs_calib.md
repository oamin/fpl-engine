# Team-GW Dixon–Coles CS / goals calibration

Seasons: 2022-23, 2023-24, 2024-25, 2025-26 · wall 45.8s

## What this is

- Unit = **team-GW side** (not FPL XI).
- **FD**: football-data opening AH+OU → Dixon–Coles `(λ_h, λ_a, ρ)` → `xcs` / `xg` (same moments as L3/L4).
- **Opta**: oracle end-of-season ranks → centreline λ → DC moments (leaky on ranks; checks whether Opta λ→CS is also low).
- Actual CS = `GA == 0`.

## Pooled FD (all seasons)

- CS (DC xcs): n=3038  mean p=0.252  actual=0.242  bias(p−y)=+0.010  brier=0.173
- CS (Poisson e^{-λ_concede}): n=3038  mean p=0.252  actual=0.242  bias(p−y)=+0.010  brier=0.173
- Goals (xG): n=3038  mean xG=1.497  GF=1.477  bias=+0.020  MAE=0.929

### FD GW8–12 only (XI-screen window)

- CS (DC xcs): n=392  mean p=0.254  actual=0.258  bias(p−y)=-0.003  brier=0.176
- CS (Poisson): n=392  mean p=0.254  actual=0.258  bias(p−y)=-0.003  brier=0.176
- Goals: n=392  mean xG=1.472  GF=1.449  bias=+0.023  MAE=0.916

## Pooled Opta centreline

- CS: n=3040  mean p=0.255  actual=0.242  bias(p−y)=+0.013  brier=0.173
- Goals: n=3040  mean xG=1.477  GF=1.477  bias=-0.000  MAE=0.939

### Opta GW8–12

- CS: n=392  mean p=0.254  actual=0.258  bias(p−y)=-0.003  brier=0.186
- Goals: n=392  mean xG=1.469  GF=1.449  bias=+0.020  MAE=0.926

## By season (FD all GWs)

| season | n | mean p_CS | actual CS | bias p−y | mean xG | GF | bias xG−GF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 760 | 0.271 | 0.272 | -0.002 | 1.416 | 1.426 | -0.010 |
| 2023-24 | 760 | 0.240 | 0.207 | +0.033 | 1.568 | 1.639 | -0.072 |
| 2024-25 | 760 | 0.237 | 0.234 | +0.002 | 1.562 | 1.467 | +0.095 |
| 2025-26 | 758 | 0.260 | 0.255 | +0.006 | 1.441 | 1.375 | +0.066 |

## FD reliability (pooled)

| p bin | n | mean p | actual CS |
| --- | ---: | ---: | ---: |
| 0.0–0.2 | 1091 | 0.134 | 0.136 |
| 0.2–0.4 | 1609 | 0.288 | 0.273 |
| 0.4–0.6 | 333 | 0.456 | 0.435 |
| 0.6–0.8 | 4 | 0.671 | 0.500 |
| 0.8–1.0 | 1 | 0.807 | 0.000 |

## Verdict

FD DC xCS bias (p−actual) pooled **+0.010**; GW8–12 **-0.003**. Goals
xG−GF pooled **+0.020**. Reliability bins track actual CS rates closely.

**Dixon–Coles is not the problem.** Team-GW P(CS) is well calibrated.
The multi-seed FPL XI CS underprint (DEF/GKP sheet rates ~50–70% vs model
p≈0.35–0.40) is **selection**: XIs stack the backs that kept sheets, not a
broken λ→xCS map. 2025-26 GW8–12 league team CS rate was ~27% with FD p≈27%.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.dixon_coles_team_cs_calib
```

