# Path-V diagnostic — dice vs share-conditional E[xP|k]

Protocol: `cascade_pathv_share_cond_xp` · N=8 · seed=7 · decision GW6 → horizon 19 · priced GW8 · objective `xi_plus_captain_ft`

**Diagnostic only** — not a gated four-season path-V promote.

- Smoke passed: **True**
- Top-15 freq Jaccard (dice ∩ share_mean): **0.667**
- Wall: 417.9s

## Micro (fixed k=2–1, Star FWD)

- Dice xp mean/var over 200 seeds: **6.825** / **7.029**
- Share_mean xp (nunique across seeds): **7.103** / nunique=1
- Share_mean is deterministic given the scoreline (player RNG off).

## Dice strands (player_attack=dice)

- Squads: 96 · strands_with_squads=8
- Mean transfers / hits: 2.875 / 7.5
- Top captain: {'name': 'Munoz', 'freq_captain': 0.3541666666666667}
- Elapsed: 210.0s

Top freq:

- Tzolakis (hull): freq=0.9375 xi=0.28125 cap=0.0
- Munoz (liverpool): freq=0.9270833333333334 xi=0.9166666666666666 cap=0.3541666666666667
- Tarkowski (everton): freq=0.84375 xi=0.8125 cap=0.09375
- Hall (newcastle): freq=0.84375 xi=0.8020833333333334 cap=0.07291666666666667
- Dasilva (coventry): freq=0.8020833333333334 xi=0.59375 cap=0.041666666666666664
- Calvert-Lewin (leeds): freq=0.75 xi=0.08333333333333333 cap=0.0
- João Pedro (chelsea): freq=0.6875 xi=0.5 cap=0.010416666666666666
- Lewis-Potter (brentford): freq=0.6770833333333334 xi=0.65625 cap=0.10416666666666667
- Tavernier (bournemouth): freq=0.65625 xi=0.625 cap=0.07291666666666667
- Rushworth (coventry): freq=0.59375 xi=0.4166666666666667 cap=0.0
- Haaland (man city): freq=0.59375 xi=0.4791666666666667 cap=0.010416666666666666
- Groß (brighton): freq=0.4895833333333333 xi=0.4583333333333333 cap=0.07291666666666667
- Hume (sunderland): freq=0.46875 xi=0.0 cap=0.0
- Saka (arsenal): freq=0.4583333333333333 xi=0.4375 cap=0.052083333333333336
- Belloumi (hull): freq=0.3854166666666667 xi=0.3541666666666667 cap=0.010416666666666666

## Share-mean strands (player_attack=share_mean)

- Squads: 96 · strands_with_squads=8
- Mean transfers / hits: 2.6818181818181817 / 6.7272727272727275
- Top captain: {'name': 'Munoz', 'freq_captain': 0.40625}
- Elapsed: 206.7s

Top freq:

- Tzolakis (hull): freq=1.0 xi=0.3125 cap=0.0
- Hall (newcastle): freq=0.9375 xi=0.875 cap=0.07291666666666667
- Tarkowski (everton): freq=0.90625 xi=0.90625 cap=0.0625
- Dasilva (coventry): freq=0.8958333333333334 xi=0.6770833333333334 cap=0.11458333333333333
- Munoz (liverpool): freq=0.8854166666666666 xi=0.875 cap=0.40625
- Tavernier (bournemouth): freq=0.78125 xi=0.75 cap=0.07291666666666667
- Calvert-Lewin (leeds): freq=0.7291666666666666 xi=0.0 cap=0.0
- João Pedro (chelsea): freq=0.7291666666666666 xi=0.625 cap=0.0
- Lewis-Potter (brentford): freq=0.7291666666666666 xi=0.7083333333333334 cap=0.08333333333333333
- Rushworth (coventry): freq=0.6875 xi=0.4479166666666667 cap=0.0
- Haaland (man city): freq=0.6666666666666666 xi=0.5833333333333334 cap=0.0
- Gonzalo (fulham): freq=0.65625 xi=0.0 cap=0.0
- Groß (brighton): freq=0.6041666666666666 xi=0.59375 cap=0.041666666666666664
- Bogle (leeds): freq=0.46875 xi=0.17708333333333334 cap=0.0
- Van Hecke (spurs): freq=0.4583333333333333 xi=0.3645833333333333 cap=0.010416666666666666

## Read

- Same rolled scorelines / Opta path; only the player layer changes.
- **Micro:** share_mean kills player-layer noise given `k` (as designed).
- **Path-V:** top-15 Jaccard **0.67** (vs ~0.36 for gen vs mean earlier); hits **7.5 → 6.7**, transfers ~flat. Milder than dice, still nowhere near mean-layer hits (~1.6).
- Verdict: keep `share_mean` as a gen path-V candidate default; **not** a gated promote.

Artifacts:
- `/workspace/data/predictions/2026-27/cascade_pathv_dice`
- `/workspace/data/predictions/2026-27/cascade_pathv_share_mean`

Reproduce:
```bash
PYTHONPATH=. python3 -m src.eval.cascade_pathv_share_cond
```
