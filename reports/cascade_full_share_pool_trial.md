# Full-club share pool trial (selection stays thin)

Season **2025-26** · N=40 · seed=7 · α_G=0.9556 · wall 182.3s

Open-loop XI from the mean `xi_ft` rolling artifact (selection ≥60′). Generative dice built on **thin** (≥60′) vs **full** (m>0, impact subs).

- Thin share pool size: **157** (City 4)
- Full share pool size: **379** (City 18)

## OL XI GW8–19 count residual

| arm | Σ E[G] | Σ G | Σ resid | mean resid / match |
| --- | ---: | ---: | ---: | ---: |
| thin_shares | 75.27 | 28.00 | +47.27 | +0.358 |
| full_shares | 39.77 | 28.00 | +11.77 | +0.089 |

Δ Σ resid (full − thin) = **-35.50**

## Haaland

| arm | Σ E[G] | Σ G | resid |
| --- | ---: | ---: | ---: |
| thin_shares | 20.38 | 10.00 | +10.38 |
| full_shares | 9.68 | 10.00 | -0.32 |

## Sample π (strand 0, GW8 first scored sides)

### thin_shares

- **sunderland** (n=7): Granit Xhaka 0.40, Omar Alderete 0.30, Noah Sadiki 0.13, Chemsdine Talbi 0.11, Nordi Mukiele 0.04
- **wolves** (n=6): Ladislav Krejcí 0.48, Marshall Munetsi 0.24, João Victor Gomes da Silva 0.20, Emmanuel Agbadou 0.07, Hugo Bueno López 0.00

### full_shares

- **sunderland** (n=19): Wilson Isidor 0.35, Enzo Le Fée 0.14, Granit Xhaka 0.12, Omar Alderete 0.09, Bertrand Traoré 0.07
- **wolves** (n=21): Jørgen Strand Larsen 0.26, Ladislav Krejcí 0.14, Hwang Hee-chan 0.12, Santiago Ignacio Bueno 0.10, Tolu Arokodare 0.08

## Verdict

**Confirmed.** Full-club shares cut OL XI ΣE[G] resid from +47.3 to **+11.8**
(Δ −35.5). Haaland resid +10.4 → −0.3. Selection pool stays ≥60′; mean
`xi_ft` hist/rolling boards now pass `_build_hist_share_pool` into generative
dice.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_full_share_pool_trial --n 40
```

