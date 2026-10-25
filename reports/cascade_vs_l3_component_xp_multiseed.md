# Multi-seed xP(G,CS,GA) residuals (boards once)

Season **2025-26** · GW [8, 9, 10, 11, 12] · N=40 · 5 seeds · wall 440.8s

## Protocol

- Build cascade + l3_v2 component boards **once** per GW.
- Several GW8 seeded 15s; **hold** each squad (no transfers).
- Each GW: seed XI at GW8 if given, else `pick_xi` on score_xp_v2.
- Residual = Σ(xP − actual) on that XI. No captain.

## Per seed

| seed | J(l3_wc) | casc rG | casc rCS | casc rGA | l3 rG | l3 rCS | l3 rGA | ΣP_G | ΣP_CS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l3_wc | 1.00 | +17.8 | -22.7 | -2.3 | +2.7 | -22.0 | -3.1 | 36 | 60 |
| v2_no_haaland | 0.58 | +34.2 | -22.2 | -2.7 | +16.7 | -22.0 | -3.4 | 16 | 60 |
| v2_no_arsenal | 0.50 | +18.2 | -21.1 | -4.1 | +1.0 | -24.8 | -4.7 | 36 | 48 |
| v2_no_liverpool | 0.67 | +18.3 | -21.9 | -3.8 | +1.6 | -25.1 | -4.5 | 36 | 62 |
| casc_ol | 0.20 | +20.4 | -6.2 | +2.1 | +12.1 | -7.0 | +1.4 | 42 | 52 |

## Cross-seed midpoint

### cascade

- r_G: mean **+21.8** (sd 6.2, range [+17.8, +34.2]) does **not** cross 0
- r_CS: mean **-18.8** (sd 6.3, range [-22.7, -6.2]) does **not** cross 0
- r_GA: mean **-2.1** (sd 2.2, range [-4.1, +2.1])

### l3_v2

- r_G: mean **+6.8** (sd 6.4, range [+1.0, +16.7]) does **not** cross 0
- r_CS: mean **-20.2** (sd 6.7, range [-25.1, -7.0]) does **not** cross 0
- r_GA: mean **-2.8** (sd 2.2, range [-4.7, +1.4])

## Verdict

**CS does not swing around zero.** Across 5 opens, casc r_CS ∈ [−22.7, −6.2]
(mean −18.8); l3_v2 ∈ [−25.1, −7.0] (mean −20.2). The least-negative seed is
`casc_ol` (different back line); every seed still underprints CS.

**G stays non-negative** on this hold-squad protocol: casc r_G ∈ [+17.8, +34.2]
(mean +21.8); l3_v2 ∈ [+1.0, +16.7] (mean +6.8). No seed crosses below 0.

So the earlier single-XI CS gap was not just L3-open luck — a shared CS
underprint remains under diverse GW8 squads (held, no transfers). GA still
straddles 0.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp_multiseed --n 40
```

