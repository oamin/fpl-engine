# Layer 5 audit — Steps 1–3

Helpers wrap `ChipWallet` / `hit_cost` / `advance_ft` / `captain_multiplier`. Step 2 = TC/BB scoring identity. Step 3 = WC keeps rebuild / FH reverts / FT preserved. Joint chip+FT foresight is Step 4. No play thresholds.

Cases passed: **27/27**

| check | pass | detail |
| --- | --- | --- |
| `gw1_no_wc_fh` | PASS | available=('bench_boost', 'triple_captain') |
| `gw1_tc_bb_ok` | PASS | available=('bench_boost', 'triple_captain') |
| `one_chip_per_week` | PASS | after TC available=() |
| `h1_tc_spent` | PASS | GW10=('wildcard', 'free_hit', 'bench_boost') |
| `h2_tc_fresh` | PASS | GW20=('wildcard', 'free_hit', 'bench_boost', 'triple_captain') |
| `h2_has_full_set` | PASS | GW20=('wildcard', 'free_hit', 'bench_boost', 'triple_captain') |
| `h2_wc_once` | PASS | GW25=('free_hit', 'bench_boost', 'triple_captain') |
| `fh_adjacent_blocked` | PASS | GW20 after H1 FH=('wildcard', 'bench_boost', 'triple_captain') |
| `fh_h2_later_ok` | PASS | GW21=('wildcard', 'free_hit', 'bench_boost', 'triple_captain') |
| `wc_zero_hits` | PASS | hits=0 |
| `fh_ft_preserved` | PASS | ft_after=2 |
| `none_ordinary_hit` | PASS | hits=8 |
| `tc_captain_x3` | PASS | tc=3 none=2 |
| `actions_prefix_none` | PASS | actions=('none', 'wildcard', 'free_hit', 'bench_boost', 'triple_captain') |
| `validate_legal_map` | PASS | ok |
| `score_none_identity` | PASS | obj=30.0 |
| `score_tc_ge_none` | PASS | tc=35.0 none=30.0 |
| `score_bb_adds_bench` | PASS | bb=35.0 bench_mu=5.0 |
| `score_hits_subtract` | PASS | obj=26.0 |
| `fh_commit_is_rebuild` | PASS | commit=('b0', 'b1')… |
| `fh_next_reverts` | PASS | next_ids0=a0 bank=50 |
| `fh_ft_and_hits` | PASS | ft=2 hits=0 |
| `fh_wallet_spent` | PASS | avail=('wildcard', 'bench_boost', 'triple_captain') |
| `wc_next_keeps_rebuild` | PASS | bank=40 |
| `wc_ft_preserved` | PASS | ft=2 hits=0 |
| `none_keeps_moves` | PASS | ids0=b0 |
| `none_ft_rolls` | PASS | ft=1 hits=0 |

## Reading

- Step 1–3 gate physics only. Step 4 = strategy (joint outer beam).
- Inner boards: priced μ_post, γ=1, unconstrained hits.

JSON: `data/processed/strategy_layer5_audit.json`
