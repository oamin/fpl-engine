# Dig: ±10 G residual swings (π∝r season map GW8–38)

Source: `cascade_vs_l3_component_xp_pi_r_gw8-38.json` (shared L3 XI, no captain).

## Verdict

The ±8–20 G/GW swings are **player-haul / XI-blank Poisson noise**, not a cascade-vs-L3 disagreement. On every big week, casc and l3 move the **same direction** vs actual (Δ casc−l3 on those weeks is only ±0.2…2.8).

| class | n GWs | Σ (casc−P) G | what it is |
| --- | ---: | ---: | --- |
| Haul weeks (`resid≤−8`) | 5 | **−60.4** | 2–3 XI players score 1–3 goals; rest blank |
| Blank-XI weeks (`P_G=0`, `resid≥+8`) | 4 | **+43.6** | whole XI scores 0 goals; model still ~10 |
| Other | 22 | +30.8 | smaller week noise |
| **Season** | 31 | **+14.0** | |

## Haul weeks (underprint)

| GW | resid | drivers (P_G → resid) |
| --- | ---: | --- |
| **20** | −19.7 | Thiago **hat-trick** 12 (−10.4); Rice brace 10 (−9.6); Thiaw 6 (−5.1) |
| **8** | −13.4 | Kroupi brace 8 (−8.0); Haaland brace 8 (−5.8); Woltemade 4 (−3.0) |
| **21** | −10.0 | Thiago brace 8 (−5.7); Keane 6 (−5.5) |
| **17** | −8.8 | Ampadu 5 (−4.9); Haaland brace 8 (−4.4) |
| **16** | −8.4 | Ekitiké brace 8 (−6.8); Haaland brace 8 (−5.2) |

Pattern: **2–3 scorers, often multi-goal**, against casc xP_G ≈ 0.4–3. Each haul is −5…−10 at the player row; the XI residual is their sum (rest ≈0).

## Blank-XI weeks (overprint)

| GW | resid | top casc xP_G still unpaid |
| --- | ---: | --- |
| **36** | +12.3 | Semenyo 4.25 (!), Thiago 1.8, Wilson 1.4 — **0 goals on XI** |
| **27** | +11.3 | Haaland 3.3, Thiago 3.1, Ekitiké 2.4 |
| **22** | +10.0 | Ekitiké 2.9, Haaland 2.4, Thiago 2.2 |
| **37** | +9.9 | Thiago 2.4 + midfield crumbs |

Pattern: model posts a normal ~9–12 G week; **actual P_G = 0 for all 11**. Residual ≈ full casc total. Casc≈l3 except GW36 Semenyo (casc 4.25 vs l3 0.98 — share spike worth a separate look, not the blank itself).

## Who accumulates |resid|

| player | Σ\|resid_G\| | Σ resid | note |
| --- | ---: | ---: | --- |
| Haaland | 66.8 | +8.4 | brace weeks under, blank weeks over; season near flat |
| Thiago | 58.4 | +2.6 | hat-trick GW20 + blanks; season near flat |
| Ekitiké | 35.6 | +0.2 | brace GW16 vs blanks |
| Bruno | 27.1 | −8.9 | mild under (actual > casc) |
| Kroupi | 22.3 | −1.7 | GW8 brace |

Premium forwards’ week-level |resid| is large; **season sums net near zero**. The ±10 GW bars are the same players alternating haul vs blank.

## Implication

Do **not** retune goal shares / club GF from these swings. They are discrete goal outcomes on a fixed XI. Season ΣG (+14 casc, +51 l3) is the calibration signal; week ±10 is expected under a Poisson goal model with ~8–10 xP_G on the XI.
