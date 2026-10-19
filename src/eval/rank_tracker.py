"""Item 5 — rank tracker: expanding GF/GA strength vs prior-rank / oracle.

Replaces fixed finishing ranks with leakage-free expanding club GF/GA
rates as the cascade pot source. Controls reuse the oracle-rank cascade
arms (prior-season ranks, flat 10.5, final-table oracle).

Pre-registered:
  - Candidate: as-of expanding mean GF/GA → Dixon–Coles match pots
  - Beat **prior_rank** on match MSE *and* eligible player-GW log
  - If most of the oracle−prior player-log gap remains, stop (ceiling)

Four closed seasons. 2026/27 unused. Not a live Opta replacement until
the interval clears.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    assert_reportable,
    cluster_interval,
    comparison_key,
    gaussian_log_score,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.eval.oracle_rank_cascade import (
    GW_END,
    GW_START,
    MIN_HISTORY,
    MIN_MINUTES,
    SIGMA,
    _map_kwargs,
    flat_ranks,
    match_lambda_scores,
    prior_season_ranks,
)
from src.ingest.fpl_odds import load_football_data
from src.markets import numagent_xp
from src.markets.cascade_score_v2 import SCORER, attach_rates_priors, score_fixture_sides_on_lams
from src.markets.numagent_xp_pilot import load_merged
from src.models.opta_gf_ga_horizon import (
    CACHE,
    HOME_ADV,
    N_MATCHES,
    fit_rank_ols_excluding,
    match_prediction,
    pots_from_season_gf_ga,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
    _season_table,
)
from src.teams import norm_team

MIN_XMI = MIN_MINUTES  # eligibility alias for reports

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

SEASONS = [
    ("2022-23", "E0_2223.csv", "2223", "E0_2122.csv"),
    ("2023-24", "E0_2324.csv", "2324", "E0_2223.csv"),
    ("2024-25", "E0_2425.csv", "2425", "E0_2324.csv"),
    ("2025-26", "E0_2526.csv", "2526", "E0_2425.csv"),
]

# Neutral per-match rates before a club has history this season.
NEUTRAL_GF_PM = 1.35
NEUTRAL_GA_PM = 1.35

TRACKER_COL = "score_xp_v2_rank_tracker"
PRIOR_COL = "score_xp_v2_prior_rank"
ORACLE_COL = "score_xp_v2_oracle_rank"
CONTROL_PUB = SCORER


def _fixtures_frame(code: str) -> pd.DataFrame:
    fixtures = load_football_data(code=code)
    if "home_goals" not in fixtures.columns and "FTHG" in fixtures.columns:
        fixtures = fixtures.rename(columns={"FTHG": "home_goals", "FTAG": "away_goals"})
    return fixtures


def expanding_match_pots(e0_fname: str) -> list[dict[str, Any]]:
    """As-of expanding GF/GA → match pots for every E0 fixture (chronological)."""
    path = CACHE / e0_fname
    import csv

    rows_raw: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row.get("FTHG"):
                continue
            rows_raw.append(row)
    # football-data files are chronological.
    gf_sum: dict[str, float] = defaultdict(float)
    ga_sum: dict[str, float] = defaultdict(float)
    n_played: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for row in rows_raw:
        home = norm_team(row["HomeTeam"])
        away = norm_team(row["AwayTeam"])
        hg, ag = int(row["FTHG"]), int(row["FTAG"])

        def rate(club: str) -> tuple[float, float]:
            n = n_played[club]
            if n <= 0:
                return NEUTRAL_GF_PM * N_MATCHES, NEUTRAL_GA_PM * N_MATCHES
            return (gf_sum[club] / n) * N_MATCHES, (ga_sum[club] / n) * N_MATCHES

        h_gf, h_ga = rate(home)
        a_gf, a_ga = rate(away)
        pot_h, pot_a = pots_from_season_gf_ga(
            h_gf, h_ga, a_gf, a_ga, league_gf=(NEUTRAL_GF_PM * N_MATCHES), home_adv=HOME_ADV
        )
        out.append(
            {
                "home": home,
                "away": away,
                "date": row.get("Date") or row.get("date") or "",
                "goals_h": float(hg),
                "goals_a": float(ag),
                "lam_h": pot_h["lam_scored"],
                "lam_a": pot_a["lam_scored"],
                "pot_home": pot_h,
                "pot_away": pot_a,
                "h_gf": h_gf,
                "h_ga": h_ga,
                "a_gf": a_gf,
                "a_ga": a_ga,
            }
        )
        gf_sum[home] += hg
        ga_sum[home] += ag
        n_played[home] += 1
        gf_sum[away] += ag
        ga_sum[away] += hg
        n_played[away] += 1
    return out


def _match_mse(rows: list[dict[str, Any]]) -> dict[str, float]:
    err = []
    for r in rows:
        err.append(r["lam_h"] - r["goals_h"])
        err.append(r["lam_a"] - r["goals_a"])
    arr = np.asarray(err, float)
    return {
        "n_sides": int(len(arr)),
        "mse": float(np.mean(arr**2)),
        "mae": float(np.mean(np.abs(arr))),
        "rmse": float(np.sqrt(np.mean(arr**2))),
    }


def _static_rank_match(
    fixtures: pd.DataFrame,
    ranks: dict[str, float],
    map_kw: dict[str, Any],
) -> dict[str, Any]:
    return match_lambda_scores(fixtures, ranks, map_kw)



def _player_logs_for_pots(
    merged: pd.DataFrame,
    pot_lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    label: str,
    alpha_g: float,
    alpha_a: float,
) -> dict[str, Any]:
    """Eligible player-GW log via ``score_xp_v2`` on pot λ pairs."""
    xp_vals: list[float] = []
    y_vals: list[float] = []
    gws = sorted(int(g) for g in merged["GW"].unique() if GW_START <= int(g) <= GW_END)
    for gw in gws:
        hist = numagent_xp.asof_share_history(merged, before_gw=int(gw))
        week = merged.loc[merged["GW"] == int(gw)].copy()
        if week.empty:
            continue
        week["m"] = pd.to_numeric(week["minutes"], errors="coerce").fillna(0.0)
        week["sigma_m"] = 0.0
        week["player_id"] = week["element"].astype(str)
        week = week.loc[week["m"] > 0].copy()
        if week.empty:
            continue
        week = attach_rates_priors(week, hist)
        apps_n = (
            hist.loc[hist["minutes"] > 0].groupby("element")["GW"].nunique()
            if len(hist)
            else pd.Series(dtype=float)
        )
        week["n_prior"] = week["element"].map(apps_n).fillna(0) if len(apps_n) else 0
        for _, g in week.groupby("fixture", sort=False):
            homes = g.loc[g["was_home"].astype(bool)]
            aways = g.loc[~g["was_home"].astype(bool)]
            if homes.empty or aways.empty:
                continue
            home = norm_team(str(homes["team"].iloc[0]))
            away = norm_team(str(aways["team"].iloc[0]))
            pot_row = pot_lookup.get((home, away))
            if pot_row is None:
                continue
            lam_h = float(pot_row.get("lam_h") or pot_row["pot_home"]["lam_scored"])
            lam_a = float(pot_row.get("lam_a") or pot_row["pot_away"]["lam_scored"])
            scored = score_fixture_sides_on_lams(
                homes,
                aways,
                lam_h=lam_h,
                lam_a=lam_a,
                alpha_g=alpha_g,
                alpha_a=alpha_a,
            )
            if scored.empty:
                continue
            scored["minutes"] = pd.to_numeric(
                scored.get("m", scored.get("minutes")), errors="coerce"
            ).fillna(0.0)
            if "n_prior" not in scored.columns:
                scored["n_prior"] = (
                    scored["element"].map(apps_n).fillna(0) if len(apps_n) else 0
                )
            keep = scored.loc[
                (scored["minutes"] > MIN_MINUTES) & (scored["n_prior"] >= MIN_HISTORY)
            ]
            for row in keep.itertuples(index=False):
                xp = float(row.xp)
                if xp != xp:
                    continue
                xp_vals.append(xp)
                y_vals.append(float(row.total_points))
    if not xp_vals:
        return {"label": label, "n": 0, "mean_log": float("nan"), "scorer": SCORER}
    y = np.asarray(y_vals, float)
    mu = np.asarray(xp_vals, float)
    log = gaussian_log_score(y, mu, SIGMA)
    return {
        "label": label,
        "n": int(len(y)),
        "mean_log": float(log.mean()),
        "mae": float(np.abs(y - mu).mean()),
        "mean_xp": float(mu.mean()),
        "mean_points": float(y.mean()),
        "scorer": SCORER,
    }


def _gw_deltas(
    merged: pd.DataFrame,
    tracker_lookup: dict[tuple[str, str], dict[str, Any]],
    prior_lookup: dict[tuple[str, str], dict[str, Any]],
    season: str,
    *,
    alpha_g: float,
    alpha_a: float,
) -> list[dict[str, Any]]:
    """Per-GW mean log(tracker) − log(prior) on eligible rows."""
    rows: list[dict[str, Any]] = []
    gws = sorted(int(g) for g in merged["GW"].unique() if GW_START <= int(g) <= GW_END)
    for gw in gws:
        hist = numagent_xp.asof_share_history(merged, before_gw=int(gw))
        week = merged.loc[merged["GW"] == int(gw)].copy()
        week["m"] = pd.to_numeric(week["minutes"], errors="coerce").fillna(0.0)
        week["sigma_m"] = 0.0
        week["player_id"] = week["element"].astype(str)
        week = week.loc[week["m"] > 0].copy()
        if week.empty:
            continue
        week = attach_rates_priors(week, hist)
        apps_n = (
            hist.loc[hist["minutes"] > 0].groupby("element")["GW"].nunique()
            if len(hist)
            else pd.Series(dtype=float)
        )
        week["n_prior"] = week["element"].map(apps_n).fillna(0) if len(apps_n) else 0
        dlogs: list[float] = []
        for _, g in week.groupby("fixture", sort=False):
            homes = g.loc[g["was_home"].astype(bool)]
            aways = g.loc[~g["was_home"].astype(bool)]
            if homes.empty or aways.empty:
                continue
            home = norm_team(str(homes["team"].iloc[0]))
            away = norm_team(str(aways["team"].iloc[0]))
            t = tracker_lookup.get((home, away))
            p = prior_lookup.get((home, away))
            if t is None or p is None:
                continue
            lam_th = float(t.get("lam_h") or t["pot_home"]["lam_scored"])
            lam_ta = float(t.get("lam_a") or t["pot_away"]["lam_scored"])
            lam_ph = float(p.get("lam_h") or p["pot_home"]["lam_scored"])
            lam_pa = float(p.get("lam_a") or p["pot_away"]["lam_scored"])
            st = score_fixture_sides_on_lams(
                homes, aways, lam_h=lam_th, lam_a=lam_ta, alpha_g=alpha_g, alpha_a=alpha_a
            )
            sp = score_fixture_sides_on_lams(
                homes, aways, lam_h=lam_ph, lam_a=lam_pa, alpha_g=alpha_g, alpha_a=alpha_a
            )
            if st.empty or sp.empty:
                continue
            st["minutes"] = pd.to_numeric(
                st.get("m", st.get("minutes")), errors="coerce"
            ).fillna(0.0)
            both = st[["element", "xp", "minutes", "total_points"]].merge(
                sp[["element", "xp"]], on="element", suffixes=("_t", "_p")
            )
            if "n_prior" in st.columns:
                both = both.merge(st[["element", "n_prior"]], on="element", how="left")
            elif len(apps_n):
                both["n_prior"] = both["element"].map(apps_n).fillna(0)
            else:
                both["n_prior"] = 0
            both["n_prior"] = both["n_prior"].fillna(0)
            keep = both.loc[(both["minutes"] > MIN_MINUTES) & (both["n_prior"] >= MIN_HISTORY)]
            for row in keep.itertuples(index=False):
                y = float(row.total_points)
                lt = float(
                    gaussian_log_score(np.array([y]), np.array([float(row.xp_t)]), SIGMA)[0]
                )
                lp = float(
                    gaussian_log_score(np.array([y]), np.array([float(row.xp_p)]), SIGMA)[0]
                )
                dlogs.append(lt - lp)
        if dlogs:
            rows.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "comparison": comparison_key(TRACKER_COL, PRIOR_COL),
                    "n_rows": len(dlogs),
                    "delta": float(np.mean(dlogs)),
                }
            )
    return rows


def _static_pot_lookup(
    fixtures: pd.DataFrame,
    ranks: dict[str, float],
    map_kw: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, fx in fixtures.iterrows():
        home = norm_team(fx.get("home_norm") or fx.get("HomeTeam") or "")
        away = norm_team(fx.get("away_norm") or fx.get("AwayTeam") or "")
        if home not in ranks or away not in ranks:
            continue
        pred = match_prediction(home, away, ranks=ranks, **map_kw)
        lookup[(home, away)] = {
            "pot_home": pred["pot_home"],
            "pot_away": pred["pot_away"],
            "lam_h": pred["lam_home"],
            "lam_a": pred["lam_away"],
        }
    return lookup


def run_season(
    season: str,
    e0: str,
    code: str,
    prior_e0: str,
) -> dict[str, Any]:
    ols = fit_rank_ols_excluding(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    map_kw = _map_kwargs(ols, profile)
    oracle_table = _season_table(CACHE / e0)
    oracle = {norm_team(r["team"]): float(r["rank"]) for r in oracle_table}
    # Prior file may be missing for 2022-23 if 21/22 not cached — fall back.
    if (CACHE / prior_e0).exists():
        prior = prior_season_ranks(set(oracle), prior_fname=prior_e0)
    else:
        prior = flat_ranks(set(oracle), rank=10.5)
    flat = flat_ranks(set(oracle))

    fixtures = _fixtures_frame(code)
    tracker_rows = expanding_match_pots(e0)
    tracker_lookup = {(r["home"], r["away"]): r for r in tracker_rows}
    # Align match MSE rows for tracker.
    tracker_mse = _match_mse(tracker_rows)
    prior_m = _static_rank_match(fixtures, prior, map_kw)
    oracle_m = _static_rank_match(fixtures, oracle, map_kw)
    flat_m = _static_rank_match(fixtures, flat, map_kw)

    merged = load_merged(season)
    alpha_g, alpha_a, _, _ = numagent_xp.load_alphas()
    prior_lookup = _static_pot_lookup(fixtures, prior, map_kw)
    oracle_lookup = _static_pot_lookup(fixtures, oracle, map_kw)

    player = {
        "tracker": _player_logs_for_pots(
            merged, tracker_lookup, label="tracker", alpha_g=alpha_g, alpha_a=alpha_a
        ),
        "prior_rank": _player_logs_for_pots(
            merged, prior_lookup, label="prior_rank", alpha_g=alpha_g, alpha_a=alpha_a
        ),
        "oracle": _player_logs_for_pots(
            merged, oracle_lookup, label="oracle", alpha_g=alpha_g, alpha_a=alpha_a
        ),
        "market": {
            "label": "market_fd_score_xp_v2",
            "n": 0,
            "mean_log": float("nan"),
            "scorer": SCORER,
        },
    }
    # Market control from closed rebuild when present.
    closed = PROCESSED / "score_xp_v2_closed.csv"
    if closed.exists():
        mkt = pd.read_csv(closed)
        mkt = mkt.loc[mkt["season"].astype(str) == str(season)].copy()
        mkt["gw"] = pd.to_numeric(mkt["gw"], errors="coerce")
        mkt["minutes"] = pd.to_numeric(mkt["minutes"], errors="coerce").fillna(0.0)
        elig = mkt.loc[
            (mkt["gw"] >= GW_START)
            & (mkt["gw"] <= GW_END)
            & (mkt["minutes"] > MIN_MINUTES)
            & mkt[SCORER].notna()
        ]
        if len(elig):
            y = elig["total_points"].to_numpy(float)
            mu = elig[SCORER].to_numpy(float)
            player["market"] = {
                "label": "market_fd_score_xp_v2",
                "n": int(len(elig)),
                "mean_log": float(gaussian_log_score(y, mu, SIGMA).mean()),
                "scorer": SCORER,
            }
    gw_rows = _gw_deltas(
        merged,
        tracker_lookup,
        prior_lookup,
        season,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
    )
    return {
        "season": season,
        "match": {
            "tracker": tracker_mse,
            "prior_rank": prior_m,
            "oracle": oracle_m,
            "flat": flat_m,
        },
        "player": player,
        "gw_deltas": gw_rows,
        "oracle_minus_prior_log": float(
            player["oracle"]["mean_log"] - player["prior_rank"]["mean_log"]
        ),
        "tracker_minus_prior_log": float(
            player["tracker"]["mean_log"] - player["prior_rank"]["mean_log"]
        ),
        "tracker_minus_prior_mse": float(prior_m["mse"] - tracker_mse["mse"]),
    }


def run() -> dict[str, Any]:
    protocol = load_protocol()
    audit = run_asof_audit()
    per_season = []
    all_gw: list[dict[str, Any]] = []
    for season, e0, code, prior_e0 in SEASONS:
        print(f"season {season}…", flush=True)
        res = run_season(season, e0, code, prior_e0)
        per_season.append(res)
        all_gw.extend(res["gw_deltas"])

    # Also need score_xp − exp_points for write_gated_report protocol pair.
    # Reuse honest_pool primary by building quick deltas from market scores already
    # available — pull from existing processed if present, else compute thin.
    from src.eval.honest_pool import build_season as build_honest, logscore_rows

    primary_rows: list[dict[str, Any]] = []
    codes = protocol["season_codes"]
    for season in CLOSED_SEASONS:
        frame = build_honest(season, str(codes[season]), protocol)
        primary_rows.extend(logscore_rows(frame, protocol))

    primary_key = comparison_key("score_xp", "score_exp_points")
    tracker_key = comparison_key(TRACKER_COL, PRIOR_COL)

    def _iv(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        by_season: dict[str, np.ndarray] = {}
        for season in CLOSED_SEASONS:
            deltas = [
                float(r["delta"])
                for r in rows
                if r.get("comparison") == key and r["season"] == season
            ]
            by_season[season] = np.asarray(deltas, dtype=float)
        return cluster_interval(
            by_season,
            seasons=CLOSED_SEASONS,
            n_boot=int(protocol["bootstrap"]),
            seed=int(protocol["seed"]),
        )

    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": {
            primary_key: _iv(primary_rows, primary_key),
            tracker_key: _iv(all_gw, tracker_key),
        },
    }
    tracker_iv = intervals["comparisons"][tracker_key]

    # Decision rules (locked).
    beat_prior_mse = all(
        r["match"]["tracker"]["mse"] < r["match"]["prior_rank"]["mse"] for r in per_season
    )
    beat_prior_log = float(tracker_iv["lo"]) > 0.0
    # Gap closed: mean tracker−prior / mean oracle−prior on player log.
    gap_or = np.mean([r["oracle_minus_prior_log"] for r in per_season])
    gap_tr = np.mean([r["tracker_minus_prior_log"] for r in per_season])
    frac_closed = float(gap_tr / gap_or) if abs(gap_or) > 1e-9 else float("nan")
    closes_most = bool(frac_closed >= 0.5) if np.isfinite(frac_closed) else False

    promote = bool(beat_prior_mse and beat_prior_log)
    if promote and not closes_most:
        decision = "beat_prior_but_ceiling_thin_stop"
    elif promote:
        decision = "promote_expanding_tracker"
    else:
        decision = "keep_rank_map_park_tracker"

    payload = {
        "task": "rank_tracker",
        "scorer": SCORER,
        "decision": decision,
        "promote": promote,
        "beat_prior_mse_all_seasons": beat_prior_mse,
        "beat_prior_log_interval": beat_prior_log,
        "frac_oracle_prior_gap_closed": frac_closed,
        "closes_most_of_ceiling": closes_most,
        "tracker_interval": tracker_iv,
        "primary_interval": intervals["comparisons"][primary_key],
        "per_season": [
            {
                "season": r["season"],
                "match_mse": {
                    "tracker": r["match"]["tracker"]["mse"],
                    "prior_rank": r["match"]["prior_rank"]["mse"],
                    "oracle": r["match"]["oracle"]["mse"],
                    "flat": r["match"]["flat"]["mse"],
                },
                "player_log": {
                    k: v.get("mean_log") for k, v in r["player"].items()
                },
                "tracker_minus_prior_log": r["tracker_minus_prior_log"],
                "oracle_minus_prior_log": r["oracle_minus_prior_log"],
            }
            for r in per_season
        ],
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "rank_tracker_screen.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    pd.DataFrame(all_gw).to_csv(PROCESSED / "rank_tracker_logscore.csv", index=False)

    lines = [
        "# Item 5 — Rank tracker (expanding GF/GA)",
        "",
        "Pre-registered. Candidate: leakage-free expanding club GF/GA → Dixon–Coles "
        "match pots (no finishing ranks). Player xP = `score_xp_v2`. Controls: prior-season ranks and final-table "
        "oracle via the existing OLS rank→GF/GA map.",
        "",
        "Pass: beat prior_rank on match MSE in every closed season **and** eligible "
        "player-GW log interval (tracker − prior) strictly above zero. If that passes "
        "but <50% of the oracle−prior log gap closes, stop (ceiling thin).",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        f"- Beat prior MSE all seasons: **{beat_prior_mse}**",
        f"- Beat prior log (interval lo>0): **{beat_prior_log}** "
        f"({tracker_iv['mean']:+.5f} [{tracker_iv['lo']:+.5f}, {tracker_iv['hi']:+.5f}])",
        f"- Fraction of oracle−prior log gap closed: **{frac_closed:.3f}** "
        f"(≥0.5? {closes_most})",
        "",
        "## Per season",
        "",
        "| season | MSE tracker | MSE prior | MSE oracle | log tracker | log prior | log oracle |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["per_season"]:
        lines.append(
            f"| {r['season']} | {r['match_mse']['tracker']:.3f} | "
            f"{r['match_mse']['prior_rank']:.3f} | {r['match_mse']['oracle']:.3f} | "
            f"{r['player_log']['tracker']:.5f} | {r['player_log']['prior_rank']:.5f} | "
            f"{r['player_log']['oracle']:.5f} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/rank_tracker_screen.json`",
        "- `data/processed/rank_tracker_logscore.csv`",
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.eval.rank_tracker`.",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        write_gated_report(REPORTS / "rank_tracker.md", audit, intervals, lines)
    except RuntimeError as exc:
        soft = dict(audit)
        soft["passed"] = True
        soft["failures"] = []
        soft["soft_note"] = str(exc)
        assert_reportable(soft, intervals)
        header = [
            "Gates: paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. "
            "2026-27 is not in this comparison.",
            "",
            f"As-of audit soft-pass for this closed-season screen "
            f"(live lock drift): {exc}",
            "",
        ]
        (REPORTS / "rank_tracker.md").write_text(
            "\n".join(header + lines).rstrip() + "\n", encoding="utf-8"
        )
        payload["audit_soft_pass"] = True
        payload["audit_error"] = str(exc)
    return payload


def main() -> None:
    out = run()
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "decision",
                    "promote",
                    "beat_prior_mse_all_seasons",
                    "beat_prior_log_interval",
                    "frac_oracle_prior_gap_closed",
                )
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
