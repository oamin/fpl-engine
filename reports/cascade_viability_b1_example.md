# B1 inspection example — Wolves vs Man City, 2022-23 GW8

Same centreline `M` (FD AH+OU fit), two aggregators. Demo seeds are
`Random(7+i)` on **this fixture only**. Overnight B1 used one `Random(7)`
advancing across all fixtures/GWs/seasons, so absolute xP differ; structure matches.

## Centreline

| | |
| --- | --- |
| Match | Wolves vs Man City (fixture 80) |
| λ | home 0.794 · away 2.371 |
| Top P(k) | 0–2 (0.119), 0–1 (0.100), 1–2 (0.094), 0–3 (0.094) |

## Mean layer (closed-form on moments)

Haaland 7.33 · De Bruyne 5.56 · Bernardo 5.07 · Cancelo 4.79 · Foden 4.47 …

## One generative seed (`seed=7`)

Rolled **k = 0–3** (p≈0.094): City CS, three City scorers.

| Player | gen xP (components) |
| --- | --- |
| De Bruyne | 11.83 (G+A + soft) |
| Stones | 9.20 (CS + A) |
| Gündogan | 8.50 (goal) |
| Haaland | 8.00 (goal) |
| Ederson | 6.17 (CS) |

## Gen MC-mean over N_MC=6 vs mean (same match)

| Player | mean_xp | gen_mc_mean | actual |
| --- | --- | --- | --- |
| Haaland | 7.33 | 4.67 | 6 |
| De Bruyne | 5.56 | 5.33 | 10 |
| Bernardo | 5.07 | 1.58 | 6 |
| Cancelo | 4.79 | 0.78 | 9 |
| Foden | 4.47 | 1.28 | 8 |

Three of six demo seeds rolled **1–0 Wolves** (low-probability under `M`), so the
MC average pulls City attack/defence down vs the closed-form mean — the lever B1
tested (noisy discrete average vs moment xP), not a longer planning horizon.

Artifacts: `data/processed/cascade_viability/b1_example_players.csv`,
`b1_example_case.json`.
