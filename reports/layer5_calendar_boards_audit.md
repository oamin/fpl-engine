# Layer 5 calendar boards audit

Live-legal boards: FD/bookie pots **only** inside `BOOKIE_DEPTH` (t:t+2). Beyond t+2 = as-of team-strength fill (no bookie). L4 priced path unchanged.

Cases: **8/8**

| check | pass | detail |
| --- | --- | --- |
| `H_ge_3` | PASS | H=8 |
| `H_reaches_or_stops` | PASS | H=8 depth=8 |
| `week0_first` | PASS | gws=(12, 13, 14, 15, 16, 17, 18, 19) |
| `asof_hist` | PASS | hist_max=11 |
| `sources_tagged` | PASS |  |
| `aligned_universe` | PASS |  |
| `no_bookie_beyond_t2` | PASS | violations=[] sources={12: 'week0', 13: 'fd_priced', 14: 'fd_priced', 15: 'strength_fill', 16: 'strength_fill', 17: 'strength_fill', 18: 'strength_fill', 19: 'strength_fill'} |
| `beyond_is_strength` | PASS | beyond={15: 'strength_fill', 16: 'strength_fill', 17: 'strength_fill', 18: 'strength_fill', 19: 'strength_fill'} |

Horizon GWs: `[12, 13, 14, 15, 16, 17, 18, 19]`
Sources: `{'12': 'week0', '13': 'fd_priced', '14': 'fd_priced', '15': 'strength_fill', '16': 'strength_fill', '17': 'strength_fill', '18': 'strength_fill', '19': 'strength_fill'}`
FD fixtures: 20 · fill: 50
Bookie depth: 3

JSON: `data/processed/layer5_calendar_boards_audit.json`
