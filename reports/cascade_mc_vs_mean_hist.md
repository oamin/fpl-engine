# Cascade MC frequency vs priced mean MILP — closed-season diagnostic

Protocol: `cascade_mc_vs_mean_hist` · season **2025-26** · decision GW6 → horizon 8–19 · N=12 · seed=7

**Diagnostic only** — strategy emergence, not a gated promote. Long arm uses **oracle post-GW38 ranks** + **LOSO prior GF/GA ± bands** (not live Opta). Not comparable to the live book/Opta season.

## Design

- **Path:** FD short on priced weeks → generative `k∼M`→attack; oracle-rank long for the rest; `xi_plus_captain_ft` per strand; frequencies.
- **Mean:** single `xi_plus_captain` MILP on discounted FD mean boards GW[8, 9] only (hard stop).
- Pool: as-of minutes > 60.0 (n_apps≥2); no FFScout.
- `lineup_cov=False` (freeze as-of minutes on gen paths).
- **Attack rates:** historical baseline (`score_xp_v2_rate_baseline_prior`).

**Rate baseline (2026-10-21):** prior-season player rG/rA (else positional) — fixes early-season Cash sG=1.0 captain pathology. Path top captain is now Haaland (0.46), not Cash.
- **Attack rates:** historical baseline — prior closed seasons per player (name key), else positional (`score_xp_v2_rate_baseline_prior`).

## Path arm

- Squads solved: **72** · strands_with_squads=12
- Mean transfers / hits: 3.95 / 7.8
- Sources: ['cascade_long_oracle_rank', 'cascade_short_priced']
- Elapsed: 89.0s

Top freq:

- Erling Haaland (man city): freq=0.986 xi=0.972 cap=0.458
- Kevin Schade (brentford): freq=0.792 xi=0.778 cap=0.139
- João Pedro Junqueira de Jesus (chelsea): freq=0.653 xi=0.583 cap=0.028
- Bryan Mbeumo (man utd): freq=0.625 xi=0.611 cap=0.042
- Martin Dúbravka (burnley): freq=0.625 xi=0.125 cap=0.000
- Bruno Borges Fernandes (man utd): freq=0.569 xi=0.569 cap=0.111
- Nordi Mukiele (sunderland): freq=0.569 xi=0.472 cap=0.000
- El Hadji Malick Diouf (west ham): freq=0.514 xi=0.000 cap=0.000
- Guglielmo Vicario (spurs): freq=0.486 xi=0.375 cap=0.000
- Yankuba Minteh (brighton): freq=0.431 xi=0.417 cap=0.028
- Toti Gomes (wolves): freq=0.375 xi=0.000 cap=0.000
- Marcos Senesi Barón (bournemouth): freq=0.361 xi=0.361 cap=0.000
- Amad Diallo (man utd): freq=0.347 xi=0.333 cap=0.042
- David Raya Martín (arsenal): freq=0.319 xi=0.236 cap=0.000
- Antoine Semenyo (bournemouth): freq=0.306 xi=0.306 cap=0.000

## Mean arm (priced t:t+1)

- Objective value: **135.52**
- Captain id: `2025-26:430`
- Elapsed: 0.4s

Squad:

- Guglielmo Vicario (spurs) [XI] · μ=8.49
- Marco Bizot (aston villa) · μ=0.22
- Nordi Mukiele (sunderland) [XI] · μ=10.03
- Marcos Senesi Barón (bournemouth) [XI] · μ=11.57
- Virgil van Dijk (liverpool) [XI] · μ=10.53
- Trevoh Chalobah (chelsea) [XI] · μ=10.01
- Marc Guéhi (crystal palace) · μ=4.26
- Dominik Szoboszlai (liverpool) [XI] · μ=10.44
- Bruno Borges Fernandes (man utd) [XI] · μ=11.05
- Moisés Caicedo Corozo (chelsea) [XI] · μ=10.25
- Martín Zubimendi Ibáñez (arsenal) · μ=9.16
- Lucas Tolentino Coelho de Lima (west ham) · μ=4.27
- Erling Haaland (man city) [XI,C] · μ=14.44
- João Pedro Junqueira de Jesus (chelsea) [XI] · μ=12.94
- Hugo Ekitiké (liverpool) [XI] · μ=11.34

## Overlap (strategy emergence)

- Jaccard path-top15 ∩ mean-15: **0.250**
- Mean picks in path top15: **6** / 15
- Path top captain: {'player_id': '2025-26:430', 'name': 'Erling Haaland', 'freq_captain': 0.4583333333333333}
- Mean captain in path top: **True**

Mean picks that are also path-stable:

- João Pedro Junqueira de Jesus (`2025-26:249`)
- Erling Haaland (`2025-26:430`)
- Bruno Borges Fernandes (`2025-26:449`)
- Guglielmo Vicario (`2025-26:565`)
- Nordi Mukiele (`2025-26:694`)
- Marcos Senesi Barón (`2025-26:72`)

**Wall-clock:** 91.6s

## Artefacts

- `data/processed/cascade_mc_vs_mean_hist/state.json`
- `data/predictions/2025-26/cascade_mc_vs_mean_hist`
- `reports/cascade_mc_vs_mean_hist.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mc_vs_mean_hist --season 2025-26 --n 12
```

