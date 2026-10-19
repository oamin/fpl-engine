# Layer 5 beam width K screen

Joint chip+transfer beam on priced μ (γ=1). K ∈ 1, 2, 4, 8, 16. Nested probe: planned Σμ vs K on a shared K=1 trajectory. Realised: full receding climb per K.

## Nested planned μ (should be non-decreasing in K)

- Probe GWs: **32**
- Monotone rate: **1.000**

## Realised climb totals by K

| K | total pts | mean planned μ |
| ---: | ---: | ---: |
| 1 | 7551 | 181.70 |
| 2 | 7677 | 185.85 |
| 4 | 7654 | 190.14 |
| 8 | 7656 | 189.77 |
| 16 | 7656 | 189.77 |

## Realised Δ vs K=1 (pooled GW)

- K=2: mean Δ **+1.016**/GW · total **+126** (n=124)
- K=4: mean Δ **+0.831**/GW · total **+103** (n=124)
- K=8: mean Δ **+0.847**/GW · total **+105** (n=124)
- K=16: mean Δ **+0.847**/GW · total **+105** (n=124)

## Verdict

- **Planned μ:** monotone in K on **32/32** probe GWs (nested search works).
- **Realised:** not linear. K=2 best vs K=1 (**+126** pooled); K=4/8/16 ≈ **+103–105**. K=8≡K=16 (beam saturated).
- Default beam width: **K=2** (or 4); do not chase K=16.

JSON: `data/processed/layer5_beam_k.json`
