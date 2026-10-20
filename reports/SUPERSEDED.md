# Superseded reports

## Numagent 1.0 (superseded 2026-10-19)

**Current product is Numagent 2.0** (`protocol.numagent_2_0`). Index: `reports/LOCKED.md`.

**In-place archive index:** [`ARCHIVE_NUMAGENT_1_0.md`](ARCHIVE_NUMAGENT_1_0.md) (path list + what must stay importable).

Numagent 1.0 was the GW6-era numeric path: live XI/captain from captured **`ep_next`**, shadow/`climb` on **`score_xp`**, `wildcard_plan` / `deadline` / `whatif` on opening pots. Protocol archive: `numagent_1_0`.

Do not use 1.0 for live picks or new screens. Stragent remains a separate string agent, not Numagent.

## Scorer — `score_xp` (superseded 2026-10-19)

**Engine primary is `score_xp_v2` (Numagent 2.0).** Protocol: `primary_scorer` / `score_xp_v2_screen.replaces_published_score_xp=true`.

`score_xp` remains importable for historical free-transfer climb replay and as the named baseline in `reports/score_xp_v2_gate.md` (v2 − score_xp **+0.0531** [+0.0453, +0.0609]). Do not extend `score_xp` climbs, and do not use it for live picks or new screens except as that named baseline. Full keep-list: [`ARCHIVE_NUMAGENT_1_0.md`](ARCHIVE_NUMAGENT_1_0.md).

Assembler for the **current** product: `src/markets/numagent_xp.py`.

---

## Stage climbs 14–34

Stages 14–34, and the later climbs that used the same pool, are not the current comparison.

The buy pool in those runs kept the players who played the gameweek being scored, and the odds path preferred closing prices. The free-transfer climb on that pool was not rerun. Do not cite those totals as the scoreboard.

The procedure audit (`reports/procedure_audit.md`) is historical gate evidence for the old `score_xp` vs expected-points pair. New player-score screens use `score_xp_v2`.

The published Gameweeks 1–5 total of 280 is a historical record of that question. It used the played-only pool and the shorter formation list. It was not recomputed.

## Stage reports in this class

- `reports/stage_14_season_climb.md`
- `reports/stage_15_season_climb_ml.md`
- `reports/stage_16_ridge_multiseason.md`
- `reports/stage_17_ridge_starters.md`
- `reports/stage_18_season_climb_budget.md`
- `reports/stage_19_season_climb_ft.md`
- `reports/stage_20_season_climb_hybrid.md`
- `reports/stage_21_season_climb_smoothed.md`
- `reports/stage_22_autosubs.md`
- `reports/stage_22_fixture_structural.md`
- `reports/stage_23_uncertainty_fdr.md`
- `reports/stage_24_xp_v2_lgbm.md`
- `reports/stage_25_xp_saves_blend.md`
- `reports/stage_26_fwd_team_fade.md`
- `reports/stage_27_play_audit.md`
- `reports/stage_28_sharpe_u.md`
- `reports/stage_29_batch.md`
- `reports/stage_30_followup.md`
- `reports/stage_31_churn.md`
- `reports/stage_32_own_value.md`
- `reports/stage_33_screen.md`
- `reports/stage_34_follow.md`

Stages 35–48 used the same pool class. Their files stay in `reports/` as the record of those questions. They are not the current comparison.

`src/models/season_climb.py` and `src/models/search_protocol.py` remain importable. Running them does not produce a reportable result.
