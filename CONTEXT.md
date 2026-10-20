# FPL Engine — archive index

**Current architecture:** **Numagent 2.0** (`protocol.numagent_2_0`) — supersedes Numagent 1.0. See [`.cursor/context.md`](.cursor/context.md) and [`reports/LOCKED.md`](reports/LOCKED.md).

This file is the **superseded stage map** (stages 0–48 pot×share / climb era). Those modules and reports stay in the repo as evidence. They are not the live scorer or strategiser.

| Era | What to read |
| --- | --- |
| **Live / locked** | `.cursor/context.md`, `reports/LOCKED.md`, `reports/numagent_2_0_method.md`, `reports/strategy_layers.md` |
| Numagent 1.0 / `score_xp` archive | `reports/ARCHIVE_NUMAGENT_1_0.md` (in-place keep-list) |
| Parked hypotheses | `.cursor/ideas.md` |
| Superseded climbs | `reports/SUPERSEDED.md` |
| Protocol | `experiments/protocol.json` |

## Stages (historical)

| Stage | Module | Report |
| --- | --- | --- |
| 0 | Clean slate | `reports/stage_0_clean.md` |
| 1 | `src.ingest` | `reports/stage_1_ingest.md` |
| 2 | `src.features.shares` | `reports/stage_2_shares.md` |
| 3 | `src.models.team_market` | `reports/stage_3_team_market.md` |
| 4 | `src.features.baseline` | `reports/stage_4_baseline.md` |
| 5 | `src.models.player_mu` | `reports/stage_5_player_mu.md` |
| 6 | `src.models.shin_only_mu` | `reports/stage_6_shin_only_mu.md` |
| 7 | `src.models.xmi_historic` | `reports/stage_7_xmi_historic.md` |
| 8 | `src.models.xgi_roll3` | `reports/stage_8_xgi_roll3.md` |
| 9 | `src.models.cs_shin` | `reports/stage_9_cs_shin.md` |
| 10 | `src.models.xmi_points_gate` | `reports/stage_10_xmi_points_gate.md` |
| 11 | `src.models.channel_trial` | `reports/stage_11_channel_trial.md` |
| 12 | `src.models.long_horizon` | `reports/stage_12_long_horizon.md` |
| 13 | `src.models.xp_engine` | `reports/stage_13_xp_engine.md` |
| 14–48 | climb / repair / crowd / GW1–5 | `reports/stage_*.md`, `reports/SUPERSEDED.md` |

Engine primary scorer is `score_xp_v2`. `score_xp` is superseded (historical climb replay / gate baseline only).

## Odds

Historical screens: football-data.co.uk. **Numagent / `score_xp_v2` live pots:** The Odds API (AH+OU+BTTS; see `.cursor/rules/odds-api-quota.mdc`). Classic `gw_lines` path may use Betfair Exchange folders.
