# Dig: what drives cascade_long G residual

Sources: decision-GW Opta π∝r map GW8–38 (player rows); Opta vs FD λ vs
actual on all fixtures GW8–38; long-horizon dig (XI Σ, same Opta stack).

## Verdict

**cascade_long’s G overprint is mostly Opta away-match λ too high, flowing
into FWD `xp_goals` under π∝r.** Not a share bug and not DNP stubs.

| layer | finding |
| --- | --- |
| Club λ | Opta mean λ − actual = **+0.15**; FD ≈ **+0.01**. Split: **away +0.30**, home **~0**. |
| Opta − FD | Σ player `casc_xP_G` **+18.5** on same XI/π∝r — almost all of the Opta−FD gap. |
| Position | Opta season resid_G: **FWD +30.5** / MID −1.9 / DEF +3.9 / GK 0 (total +32.5). |
| Venue (played) | Away Σresid_G **+24.7**; home **+1.0**. Away FWD alone **+29.9**. |
| Blanks vs hauls | Non-scoring player-GWs +220; scoring −188 (Poisson mean vs haul). Net positive, **shared with FD**; not the Opta−FD gap. |
| DNP | +11.4 of +32.5 — minor. |

`HOME_ADV = 1.0` (1.08 retired PI 2026-10-24) leaves away λ unsuppressed vs the
book; FD embeds real home advantage.

## Club λ (GW8–38, N=40 Opta draws / fixture)

| | mean λ | − actual | MAE |
| --- | ---: | ---: | ---: |
| FD | 1.34 | **+0.01** | 0.89 |
| Opta | 1.48 | **+0.15** | 0.94 |
| actual | 1.33 | — | — |

| venue | Opta − act | FD − act | Opta − FD |
| --- | ---: | ---: | ---: |
| Home | 0.00 | +0.01 | −0.01 |
| Away | **+0.30** | +0.01 | **+0.28** |

## Player / position (decision-GW Opta vs FD, same L3 XI, π∝r)

- Season Σ resid_G: Opta **+32.5**, FD **+14.0** → Opta−FD **+18.5**.
- That +18.5 is **FWD +15.8**, MID +3.0, DEF −0.3.
- Largest Opta−FD casc lifts: Ekitiké, Welbeck, Kroupi, Haaland (attackers).
- Away FWD Σ(opta−fd casc) **+8.6** vs home FWD **+4.5**.

Blank/haul asymmetry is large on both arms; it does not explain why Opta sits
~18 G above FD on the same XI.

## Link to horizon dig

Long-horizon per-GW mean resid_G **~+3.5** (t+2:end) uses the same
`sample_opta_scenario` + π∝r stack. Short FD overlay on the same XI is
**~+1.9**. The gap matches the decision-GW Opta−FD story: **λ venue**, not
horizon length.

## What not to do

- Do not retune π∝r shares from this resid (FWD concentration is expected;
  Opta−FD holds shares fixed).
- Do not re-introduce `HOME_ADV=1.08` casually — it was retired as unapproved
  spice; any venue fix needs a **pre-registered** λ screen (Opta±venue vs
  actual / vs FD) before a promote.
- Haul/blank ±10 week noise is separate (already dug).

## Reproduce

```bash
# player map (existing)
PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp \
  --n 40 --goal-weight rate --match-source opta --steps 8,...,38 --decision-gw-only
# λ table: see agent dig → /tmp/opta_fd_lam_vs_act.csv
```
