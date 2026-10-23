# Live cascade layers — pool vs minutes vs scenario

PI lock 2026-10-21: generative paths use **frozen as-of mean minutes**
(`lineup_cov=False`). FFScout is a different mechanic.

## Three layers (do not conflate)

| Layer | Question | Live source | Backtest source |
| --- | --- | --- | --- |
| **Pool** | WHO may enter the squad / MILP? | FFScout firm ∩ sticky expanding-minutes gate (`player_pool`) | Same rule as-of (or hist minutes>60 ∩ AH+OU where locked) |
| **Minutes** | HOW LONG do they play if selected? | As-of mean `m` / live `xmi`; optional FPL `chance_of_playing` overlay | As-of expanding mean minutes on the side table |
| **Scenario** | WHAT happens in the match? | Book → `M` → `k∼M` → attack dice → CS/GC | FD / Opta → same stack |

## Live season

1. **Refresh pool snapshot** when FFScout/news warrants — sticky season ids for
   planning; does **not** re-roll minutes per future GW inside a plan.
2. **Build side table** with as-of rates and mean minutes / `xmi` for pool
   players (and any scored universe the strand needs).
3. **Gen paths:** keep those minutes fixed (`lineup_cov=False`); only scoreline
   and attack/CS/GC vary.
4. **Availability:** injury/doubt scales `m` via overlay (or drops from pool via
   FFScout firm cut) — still not a minutes Monte Carlo.

## What FFScout is not

- Not a draw from historical joint minutes.
- Not a substitute for as-of mean / `xmi` on the board.
- Firm XI / nailed_on informs **eligibility** (and later a firm-XI override if
  locked); playing time for scoring stays on the minutes layer.
