# Item 1a — Captain-aware multistage objective

Diagnostic comparison on the same MC seed (`--n 10 --snapshot-every 2 --seed 7`).
Not a gated likelihood promotion. Transfers not yet in the objective.

## Objectives

| Mode | What the MILP maximises |
| --- | --- |
| `squad_sum` | Σ score over all 15 (bench = full weight) |
| `xi_plus_captain` | Σ score over a legal XI **+ one captain double** (`c ≤ y ≤ x`) |

## Result (60 solved squads each)

| Player | £ | freq (`squad_sum`) | freq (`xi+cap`) | freq_captain |
| --- | ---: | ---: | ---: | ---: |
| Saka | 9.6 | 0.83 | **0.85** | **0.48** |
| Rogers | 7.8 | 0.70 | 0.73 | 0.12 |
| Haaland | 15.6 | 0.62 | 0.55 | 0.12 |
| Barry | 5.7 | 0.78 | 0.53 | 0.00 |
| Mbeumo | 7.9 | 0.62 | 0.53 | 0.10 |

Haaland stays ~**#7** under both. Captain mode shifts capital toward **Saka as the armband** (48% of captaincies) rather than forcing the £15.6 premium. Cheap enablers (Barry) drop when the objective stops paying full weight for bench.

## Read

1. Captaincy alone does **not** make Haaland transformative — MID goal pts + price efficiency still win the armband race.
2. The objective change **does** matter: bench discounting and captain concentration move frequencies.
3. Next inside item 1: **transfers / hits** on the strand (path-dependent FT bank), then chips.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 10 --snapshot-every 2 --seed 7 --objective squad_sum
PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 10 --snapshot-every 2 --seed 7 --objective xi_plus_captain \
  --out data/predictions/2026-27/multistage_xi_captain
```
