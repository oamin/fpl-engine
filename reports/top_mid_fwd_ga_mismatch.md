# Top MID+FWD G/A mismatch dig (2025-26, GW8–19)

Diagnostic on full-side minutes-oracle `score_team_side` (as-of rates/α at decision GW6).
Top = xp decile among MID+FWD with actual minutes > 0. α_G≈0.960, α_A≈0.919.

## Headline

| Scope | goals/E[G] | assists/E[A] |
| --- | ---: | ---: |
| All MID+FWD | 0.879 | 0.794 |
| **Top xp decile** | **0.851** | **0.612** |
| Rest | 0.888 | 0.842 |

## Where the miss sits (pool-relative)

Use **pool** team G/A = goals/assists by players who received shares that GW (apples-to-apples with `G_team`/`A_team`). ~17 goals on top team-weeks were by players **outside** the frozen GW6 side table (e.g. Junior Kroupi) — do not count those in share checks.

| Channel | pot = team_pool / pot | if shares perfect, expect player ratio | actual player ratio | share factor |
| --- | ---: | ---: | ---: | ---: |
| Goals | **0.823** | 0.823 | 0.851 | **1.034** (OK / slightly under-share) |
| Assists | **0.717** | 0.717 | 0.612 | **0.854** (over-share on top) |

### Goals — mostly pot

- Unconditional pot on top fixtures is short (~18%): high-xG sides blank often enough to drag E[G].
- Blank team-goal rows: 42/174; those rows carry 16.7 of top E[G] with 0 goals.
- When the team *does* score (n=132): `sG`≈0.220 vs realised pool share ≈0.234; goals vs `sG·pool_g` ≈ **1.035**. Shares are fine; Haaland-class blanks are variance on a fat pot.
- FWD g/EG≈0.855, MID≈0.848 — same story.

### Assists — pot **and** share

- Pot short (~28%): `A_team` overstates assists on the same high-μ fixtures.
- Conditional on team assists > 0: `sA`≈0.161 vs realised ≈0.139 (**1.16×** too fat on top); delivery vs `sA·pool_a` ≈ **0.858**.
- Same team-weeks: top rows a/EA≈0.612 vs other players on those sides ≈0.838 — assist mass is over-allocated to the hot μ tail.
- FWD assists worse (a/EA≈0.534) than MID (≈0.645).

Realised MID+FWD assist/goal shares are much more winner-take-most than predicted (HHI_real≈0.63 vs HHI_pred≈0.12–0.17), but the **top-xp** player is often not the week’s assister — so flattening-vs-reality does not rescue top-μ assist calibration.

## Named regulars (n≥4 in top decile)

**Goals OK-ish, assists empty:** João Pedro (9×: g/EG 0.76, a/EA **0**), Bowen (8×: 0.87 / **0**), Mbeumo (4×: 0.67 / **0**).

**Goals fine, assists short:** Haaland (12×: 1.12 / 0.78), Foden (12×: 1.39 / **0.48**), Ekitiké (7×: 1.15 / 0.55), Semenyo (8×: 1.21 / 0.51).

**Both short:** Szoboszlai (8×: 0.47 / 0.41).

Largest single-GW goal blanks are almost all premium FWDs on fat pots who scored 0 while the side sometimes still scored (Haaland GW13 City 3, João Pedro GW8 Chelsea 3, etc.).

## Implication

Residual mean bias from selecting the hot μ tail is **not** mainly “wrong goal shares on scorers.” It is:

1. **Goal pots** too high on the fixtures those players sit in (plus a few mid-season attackers missing from the GW6 share roster).
2. **Assist pots** too high **and** **assist shares** too concentrated on the same top-μ names.
