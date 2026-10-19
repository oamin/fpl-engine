# Audit: FT bank, hits, chip halves (2026/27)

Source of truth in-repo: `src/rules/fpl_2026.py`.  
Checked against Premier League FPL rules pages for 2026/27 (transfers + chips).

| Rule | Code | Official | Match |
| --- | --- | --- | --- |
| Hit cost | `HIT_COST = 4` | −4 per transfer beyond bank | ✅ |
| FT grant / week | `advance_ft` → `remaining + 1` | 1 free transfer each GW after first deadline | ✅ |
| FT bank cap | `MAX_FT = 5` | max 5 stored | ✅ |
| Cap waste | `min(MAX_FT, remaining + 1)` | 6th not added | ✅ |
| Hits empty bank | extras → `remaining = 0` then +1 | bank spent, then next GW grant | ✅ |
| WC/FH keep bank | no spend; no +1 grant | saved FTs retained; chip week does not add another | ✅ |
| Chip set | WC, FH, BB, TC | same four | ✅ |
| Per half | `ChipWallet` + `FIRST_HALF_END_GW = 19` | H1 through GW19 deadline; H2 from GW20 | ✅ |
| H1 no carry | half key on play | unused H1 chips cannot be used in H2 | ✅ |
| One chip / GW | `played_gw` | at most one chip per GW | ✅ |
| GW1 | WC/FH blocked | WC/FH not in GW1; BB/TC OK | ✅ |
| FH adjacency | no FH if \|Δgw\|=1 to prior FH | no FH in GW20 if FH in GW19 | ✅ |

**Verdict:** FT and chip expiry rules match 2026/27. Safe base for FT-DP / chip-stopping arms.

No climb. No parameter change.
