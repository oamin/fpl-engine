# Where the xP(G,CS,GA) − actual gap comes from

Same shared XI panel as `cascade_vs_l3_component_xp` (GW8 = L3/L4 WC XI;
GW9–12 = L3 owned + `pick_xi` on v2). N=40. No captain.

Residual = xP − actual. Numbers below are Σ over 55 XI player-GWs unless noted.

## Headline

| arm | Σ r_G | Σ r_CS | Σ r_GA |
| --- | ---: | ---: | ---: |
| cascade | **+9.8** | **−25.6** | +0.2 |
| l3_v2 | −4.3 | −27.7 | −0.5 |

GA is fine. The story is **CS underprint** (both arms) and a **G cancel**
(cascade: blanks overprint + scorers underprint → small net over; l3: same
shape, net slight under).

## 1. Clean sheets dominate the gap

- Actual CS rate on GKP+DEF with m≥60: **73%** (16/22).
- Mean implied p(CS) ≈ xP_CS/4: casc **0.44**, l3 **0.42**.
- So both models are roughly league-reasonable (~40–45%); this XI’s back line
  just **kept a lot of sheets**.

By GW (DEF+GKP m≥60 CS rate vs mean p_casc):

| GW | actual CS rate | mean p_casc | mean p_l3 |
| --- | ---: | ---: | ---: |
| 8 | **1.00** (6/6) | 0.28 | 0.26 |
| 9 | 0.75 | 0.38 | 0.47 |
| 10 | **1.00** (4/4) | 0.58 | 0.48 |
| 11 | 0.25 | 0.54 | 0.53 |
| 12 | 0.50 | 0.51 | 0.44 |

GW8 alone is ~70% of cascade’s Σ r_CS (−18 of −26): six clean sheets at ~28%
prior.

Split on whether CS points landed:

| | casc r_CS | l3 r_CS |
| --- | ---: | ---: |
| CS happened (P_CS>0) | −39.8 | −39.5 |
| no CS | +14.1 | +12.4 |
| **net** | **−25.6** | **−27.7** |

Almost all of the CS shortfall sits on **DEF** (casc −24.2 / l3 −25.1).
GKP is nearly flat; MID CS is noise.

## 2. Goals: blanks vs scorers (not a single bias)

Cascade Σ r_G = +9.8 hides two large opposite terms:

| slice | casc r_G | l3 r_G |
| --- | ---: | ---: |
| scorers (P_G>0, n=8) | **−29.4** | −31.1 |
| blanks (P_G=0) | **+39.1** | +26.7 |
| **net** | **+9.8** | **−4.3** |

By position:

| pos | P_G | casc xP_G | casc r_G | l3 xP_G | l3 r_G |
| --- | ---: | ---: | ---: | ---: | ---: |
| FWD | 44 | 30.0 | **−14.0** | 21.5 | **−22.5** |
| MID | 0 | 13.2 | **+13.2** | 10.5 | +10.5 |
| DEF | 0 | 10.5 | +10.5 | 7.7 | +7.7 |

FWDs under-forecast (Haaland / Kroupi / Welbeck hauls); MID+DEF spray ~1–2
xP_G per blank week that never pays. Cascade is **more aggressive on blanks**
than l3 (+39 vs +27), so it ends net-positive on G while l3 is slightly shy.

Implied FWD goals over the window: actual **11**; casc E≈**7.5**; l3 E≈**5.4**.

## 3. GA is not the problem

Σ r_GA ≈ 0 for both. Per-match MAE ~0.13. Leave it alone.

## 4. Biggest player-level drivers (cascade)

| player | role in gap |
| --- | --- |
| Chalobah (5 GWs) | CS under (−11.3) + small G over on blanks |
| Sarr (5) | G blanks over (+7.6); almost no CS story |
| Haaland (5) | G under on hauls (−6.5) |
| Kroupi (5) | G under when he scored (−4.7); GW8 0 xP vs 8 P_G |
| Ballard / Nunes (GW8) | CS under on a 100% sheet week |

## Takeaway

1. **CS gap is mostly outcome luck on this XI** (73% sheets vs ~43% model),
   concentrated in GW8 (and GW10), not a broken GA stack — GA residuals are flat.
2. **G gap is allocation**: both arms under-call the FWD hauls and over-call
   MID/DEF blank goal xP; cascade’s blank overprint is larger, so net G looks
   “too high” while l3 looks “a bit low.”
3. Closing the **net** G residual is the wrong target; the useful levers are
   FWD haul rates / minutes on the likes of Kroupi, and whether MID/DEF goal
   xP should be thinner — not a global G multiplier.

```bash
PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp --n 40
```
