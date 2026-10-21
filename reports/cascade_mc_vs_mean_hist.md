# Cascade MC frequency vs priced mean MILP — closed-season diagnostic

Protocol: `cascade_mc_vs_mean_hist` · season **2025-26** · decision GW6 → horizon 8–19 · N=12 · seed=7

**Diagnostic only** — strategy emergence, not a gated promote. Long arm uses **oracle post-GW38 ranks** + **LOSO prior GF/GA ± bands** (not live Opta). Not comparable to the live book/Opta season.

## Design

- **Path:** FD short on priced weeks → generative `k∼M`→attack; oracle-rank long for the rest; `xi_plus_captain_ft` per strand; frequencies.
- **Mean:** single `xi_plus_captain` MILP on discounted FD mean boards GW[8, 9] only (hard stop).
- Pool: as-of minutes > 60.0 (n_apps≥2); no FFScout.
- `lineup_cov=False` (freeze as-of minutes on gen paths).

## Path arm

- Squads solved: **72** · strands_with_squads=12
- Mean transfers / hits: 3.533333333333333 / 6.133333333333334
- Sources: ['cascade_long_oracle_rank', 'cascade_short_priced']
- Elapsed: 87.5s

Top freq:

- Matty Cash (aston villa): freq=1.000 xi=0.986 cap=0.514
- Erling Haaland (man city): freq=0.986 xi=0.986 cap=0.153
- Bruno Borges Fernandes (man utd): freq=0.847 xi=0.847 cap=0.153
- Antoine Semenyo (bournemouth): freq=0.792 xi=0.792 cap=0.042
- Anton Stach (leeds): freq=0.708 xi=0.597 cap=0.042
- Bruno Guimarães Rodriguez Moura (newcastle): freq=0.639 xi=0.611 cap=0.056
- Jurriën Timber (arsenal): freq=0.625 xi=0.611 cap=0.000
- Martin Dúbravka (burnley): freq=0.611 xi=0.028 cap=0.000
- El Hadji Malick Diouf (west ham): freq=0.556 xi=0.069 cap=0.000
- Robin Roefs (sunderland): freq=0.514 xi=0.389 cap=0.000
- Riccardo Calafiori (arsenal): freq=0.514 xi=0.417 cap=0.000
- Trevoh Chalobah (chelsea): freq=0.500 xi=0.389 cap=0.000
- Marcos Senesi Barón (bournemouth): freq=0.486 xi=0.375 cap=0.000
- Hugo Ekitiké (liverpool): freq=0.486 xi=0.278 cap=0.000
- Nick Pope (newcastle): freq=0.444 xi=0.347 cap=0.000

## Mean arm (priced t:t+1)

- Objective value: **142.03**
- Captain id: `2025-26:430`
- Elapsed: 0.3s

Squad:

- Guglielmo Vicario (spurs) [XI] · μ=8.52
- Marco Bizot (aston villa) · μ=0.21
- Marcos Senesi Barón (bournemouth) [XI] · μ=11.49
- Virgil van Dijk (liverpool) [XI] · μ=10.10
- Trevoh Chalobah (chelsea) [XI] · μ=10.45
- Cristian Romero (spurs) · μ=7.99
- Trai Hume (sunderland) · μ=3.77
- Moisés Caicedo Corozo (chelsea) [XI] · μ=12.24
- Bruno Borges Fernandes (man utd) [XI] · μ=11.55
- Anton Stach (leeds) [XI] · μ=10.43
- Antoine Semenyo (bournemouth) [XI] · μ=10.57
- Yasin Ayari (brighton) · μ=7.87
- Erling Haaland (man city) [XI,C] · μ=15.15
- João Pedro Junqueira de Jesus (chelsea) [XI] · μ=13.77
- Hugo Ekitiké (liverpool) [XI] · μ=12.62

## Overlap (strategy emergence)

- Jaccard path-top15 ∩ mean-15: **0.304**
- Mean picks in path top15: **7** / 15
- Path top captain: {'player_id': '2025-26:36', 'name': 'Matty Cash', 'freq_captain': 0.5138888888888888}
- Mean captain in path top: **True**

Mean picks that are also path-stable:

- Trevoh Chalobah (`2025-26:226`)
- Erling Haaland (`2025-26:430`)
- Bruno Borges Fernandes (`2025-26:449`)
- Anton Stach (`2025-26:660`)
- Hugo Ekitiké (`2025-26:661`)
- Marcos Senesi Barón (`2025-26:72`)
- Antoine Semenyo (`2025-26:82`)

**Wall-clock:** 89.4s

## Artefacts

- `data/processed/cascade_mc_vs_mean_hist/state.json`
- `data/predictions/2025-26/cascade_mc_vs_mean_hist`
- `reports/cascade_mc_vs_mean_hist.md`

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_mc_vs_mean_hist --season 2025-26 --n 12
```

