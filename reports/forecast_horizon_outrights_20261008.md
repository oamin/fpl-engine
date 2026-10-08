# Forecast beyond GW7 (2026-10-08)

Gemini ([forecast horizon](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): KEEP the live H=3 window past the last liquid MATCH_ODDS week; DROP any expectation that Exchange MATCH_ODDS will price GW8+ usefully before those deadlines; KEEP coarse outright rankings (κ=0.5) as the stand-in for unpriced weeks; CHANGE only by refusing to widen the published horizon without a separate screen.

## Answers

| Question | Answer |
| --- | --- |
| Can forecast extend past GW7? | Yes inside the locked H=3 club steps (e.g. GW6–8). The half planner already walks those weeks. Do not widen published H without a historical screen. Wildcard dry-run stays at 4 steps, advisory only. |
| Match odds beyond GW7 on the Exchange? | Not at useful liquidity. Mac pull: GW6 priced; GW7 mostly tier-3 / neutral prior. GW8+ at the GW6 deadline is unquoted for decision use. Catalogue has no date filter; books are still thin. |
| Coarse rankings instead? | Already wired. `outrights_ranks.json` → `strength_match_pots` at κ=0.5 for fixture weeks not in `priced_weeks`. Prefer that over zeros or copying the last opponent. |

## Edge fix

A club missing from the 19-row outright table now defaults to E[rank]=18.5 (strength ≈ −0.84), not mid-table 0.0.
