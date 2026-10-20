"""Oracle-rank cascade diagnostic on a closed season (default 2025/26).

Uses that season's **final table** as predicted ranks (oracle), maps rank →
season GF/GA with OLS fit on **prior** closed seasons only, then scores the
Opta λ cascade → Dixon–Coles ``M`` → ``numagent_xp.score_team_side``
(``score_xp_v2``) against realised match goals and player points.

Controls: prior-season ranks (promoted clubs → 18/19/20), flat mid-table
rank 10.5, and published ``score_xp_v2`` on FD pots (closed rebuild).

This is a retrospective ceiling, not a live method and not a promote gate.
2026/27 is not used.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import gaussian_log_score
from src.ingest.fpl_odds import load_football_data
from src.markets import numagent_xp
from src.markets.cascade_score_v2 import side_moments_from_lams
from src.markets.numagent_xp_pilot import load_merged
from src.models.multistage_mc_milp import sample_scoreline
from src.models.opta_gf_ga_horizon import (
    CACHE,
    fit_rank_ols_excluding,
    gf_ga_from_rank,
    league_avg_gf,
    match_prediction,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
    _season_table,
)
from src.teams import norm_team

SCORER = numagent_xp.SCORE_NAME
MIN_MINUTES = 60.0

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

TARGET = {
    "season": "2025-26",
    "code": "2526",
    "e0": "E0_2526.csv",
    "label": "2025/26",
}
PRIOR_E0 = "E0_2425.csv"
GW_START = 5
GW_END = 38
MIN_HISTORY = 3
SIGMA = 3.0
N_MC = 40
MC_SEED = 0


def prior_season_ranks(oracle_clubs: set[str], prior_fname: str = PRIOR_E0) -> dict[str, float]:
    """Last season's ranks; promoted clubs get 18/19/20 by name order."""
    prior = season_final_ranks(prior_fname)
    ranks = {c: prior[c] for c in oracle_clubs if c in prior}
    promoted = sorted(oracle_clubs - set(prior))
    for i, club in enumerate(promoted):
        ranks[club] = float(18 + i)
    return ranks


def flat_ranks(oracle_clubs: set[str], rank: float = 10.5) -> dict[str, float]:
    return {c: float(rank) for c in oracle_clubs}


def _map_kwargs(ols: dict[str, Any], profile: dict[int, dict[str, float]]) -> dict[str, Any]:
    return {
        "profile": profile,
        "method": "ols",
        "ols_gf": tuple(ols["ols_gf"]),
        "ols_ga": tuple(ols["ols_ga"]),
        "mae_gf": float(ols["mae_gf"]),
        "mae_ga": float(ols["mae_ga"]),
        "league_gf": league_avg_gf(profile),
    }


def club_season_errors(
    oracle_table: list[dict[str, Any]],
    ranks: dict[str, float],
    ols: dict[str, Any],
    profile: dict[int, dict[str, float]],
) -> dict[str, Any]:
    """Predicted season GF/GA from ranks vs realised finals."""
    rows = []
    for r in oracle_table:
        club = norm_team(r["team"])
        pred = gf_ga_from_rank(
            ranks[club],
            profile=profile,
            method="ols",
            ols_gf=tuple(ols["ols_gf"]),
            ols_ga=tuple(ols["ols_ga"]),
            mae_gf=float(ols["mae_gf"]),
            mae_ga=float(ols["mae_ga"]),
        )
        rows.append(
            {
                "club": club,
                "rank_used": ranks[club],
                "actual_rank": r["rank"],
                "pred_gf": pred["gf"],
                "actual_gf": r["gf"],
                "pred_ga": pred["ga"],
                "actual_ga": r["ga"],
                "err_gf": pred["gf"] - r["gf"],
                "err_ga": pred["ga"] - r["ga"],
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "mae_gf": float(frame["err_gf"].abs().mean()),
        "mae_ga": float(frame["err_ga"].abs().mean()),
        "rmse_gf": float(np.sqrt((frame["err_gf"] ** 2).mean())),
        "rmse_ga": float(np.sqrt((frame["err_ga"] ** 2).mean())),
        "corr_gf": float(frame["pred_gf"].corr(frame["actual_gf"])),
        "corr_ga": float(frame["pred_ga"].corr(frame["actual_ga"])),
        "rows": rows,
    }


def match_lambda_scores(
    fixtures: pd.DataFrame,
    ranks: dict[str, float],
    map_kw: dict[str, Any],
) -> dict[str, Any]:
    """Per-fixture λ vs FTHG/FTAG."""
    rows = []
    for _, fx in fixtures.iterrows():
        home = norm_team(fx["home_norm"] if "home_norm" in fx and pd.notna(fx["home_norm"]) else fx.get("HomeTeam", ""))
        away = norm_team(fx["away_norm"] if "away_norm" in fx and pd.notna(fx["away_norm"]) else fx.get("AwayTeam", ""))
        if not home or not away:
            continue
        if home not in ranks or away not in ranks:
            continue
        hg = fx.get("home_goals", fx.get("FTHG"))
        ag = fx.get("away_goals", fx.get("FTAG"))
        if pd.isna(hg) or pd.isna(ag):
            continue
        pred = match_prediction(home, away, ranks=ranks, **map_kw)
        rows.append(
            {
                "home": home,
                "away": away,
                "lam_h": pred["lam_home"],
                "lam_a": pred["lam_away"],
                "goals_h": float(hg),
                "goals_a": float(ag),
                "err_h": pred["lam_home"] - float(hg),
                "err_a": pred["lam_away"] - float(ag),
                "lam_h_lo": pred["lam_home_lo"],
                "lam_h_hi": pred["lam_home_hi"],
                "lam_a_lo": pred["lam_away_lo"],
                "lam_a_hi": pred["lam_away_hi"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n": 0}
    sq = np.concatenate([frame["err_h"].to_numpy() ** 2, frame["err_a"].to_numpy() ** 2])
    abs_e = np.concatenate([frame["err_h"].abs().to_numpy(), frame["err_a"].abs().to_numpy()])
    # Coverage: actual goals inside λ band.
    cov_h = ((frame["goals_h"] >= frame["lam_h_lo"]) & (frame["goals_h"] <= frame["lam_h_hi"])).mean()
    cov_a = ((frame["goals_a"] >= frame["lam_a_lo"]) & (frame["goals_a"] <= frame["lam_a_hi"])).mean()
    return {
        "n_fixtures": int(len(frame)),
        "n_sides": int(len(sq)),
        "mse": float(sq.mean()),
        "mae": float(abs_e.mean()),
        "rmse": float(np.sqrt(sq.mean())),
        "band_coverage": float(0.5 * (cov_h + cov_a)),
        "mean_lam": float(np.concatenate([frame["lam_h"], frame["lam_a"]]).mean()),
        "mean_goals": float(np.concatenate([frame["goals_h"], frame["goals_a"]]).mean()),
    }


def mc_match_calibration(
    fixtures: pd.DataFrame,
    ranks: dict[str, float],
    map_kw: dict[str, Any],
    *,
    n_strands: int = N_MC,
    seed: int = MC_SEED,
) -> dict[str, Any]:
    """Sample scorelines inside Opta bands; compare strand-mean goals to actual."""
    rng = random.Random(seed)
    abs_err_mean = []
    abs_err_lam = []
    for _, fx in fixtures.iterrows():
        home = norm_team(fx.get("home_norm") or fx.get("HomeTeam") or "")
        away = norm_team(fx.get("away_norm") or fx.get("AwayTeam") or "")
        if home not in ranks or away not in ranks:
            continue
        hg = fx.get("home_goals", fx.get("FTHG"))
        ag = fx.get("away_goals", fx.get("FTAG"))
        if pd.isna(hg) or pd.isna(ag):
            continue
        pred = match_prediction(home, away, ranks=ranks, **map_kw)
        samples_h = []
        samples_a = []
        for _ in range(n_strands):
            sh, sa, _, _ = sample_scoreline(
                pred["lam_home"],
                pred["lam_away"],
                lam_h_lo=pred["lam_home_lo"],
                lam_h_hi=pred["lam_home_hi"],
                lam_a_lo=pred["lam_away_lo"],
                lam_a_hi=pred["lam_away_hi"],
                rng=rng,
            )
            samples_h.append(sh)
            samples_a.append(sa)
        mean_h = float(np.mean(samples_h))
        mean_a = float(np.mean(samples_a))
        abs_err_mean.append(abs(mean_h - float(hg)))
        abs_err_mean.append(abs(mean_a - float(ag)))
        abs_err_lam.append(abs(pred["lam_home"] - float(hg)))
        abs_err_lam.append(abs(pred["lam_away"] - float(ag)))
    return {
        "n_strands": n_strands,
        "n_sides": len(abs_err_mean),
        "mae_strand_mean": float(np.mean(abs_err_mean)) if abs_err_mean else float("nan"),
        "mae_centreline_lam": float(np.mean(abs_err_lam)) if abs_err_lam else float("nan"),
    }



def player_cascade_scores(
    merged: pd.DataFrame,
    ranks: dict[str, float],
    map_kw: dict[str, Any],
    *,
    label: str,
    alpha_g: float,
    alpha_a: float,
) -> dict[str, Any]:
    """Eligible player-GW log score of cascade ``score_xp_v2`` vs points."""
    xp_vals: list[float] = []
    y_vals: list[float] = []
    gws = sorted(int(g) for g in merged["GW"].unique() if GW_START <= int(g) <= GW_END)
    for gw in gws:
        hist = numagent_xp.asof_share_history(merged, before_gw=int(gw))
        # Need ≥ MIN_HISTORY appearance GWs for eligibility later.
        week = merged.loc[merged["GW"] == int(gw)].copy()
        if week.empty or hist.empty:
            continue
        rates = numagent_xp.player_rates_from_history(hist, id_col="element")
        priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
        # Appearance count as-of
        apps = (
            hist.loc[hist["minutes"] > 0]
            .groupby("element")["GW"]
            .nunique()
            .rename("n_prior")
        )
        # Fixtures: one home team + away team per fixture id
        for fid, g in week.groupby("fixture", sort=False):
            homes = g.loc[g["was_home"].astype(bool)]
            aways = g.loc[~g["was_home"].astype(bool)]
            if homes.empty or aways.empty:
                continue
            home = norm_team(str(homes["team"].iloc[0]))
            away = norm_team(str(aways["team"].iloc[0]))
            if home not in ranks or away not in ranks:
                continue
            try:
                pred = match_prediction(home, away, ranks=ranks, **map_kw)
            except KeyError:
                continue
            mom_h = side_moments_from_lams(pred["lam_home"], pred["lam_away"])
            mom_a = side_moments_from_lams(pred["lam_away"], pred["lam_home"])
            for side_df, mom in ((homes, mom_h), (aways, mom_a)):
                players = side_df.copy()
                players["m"] = pd.to_numeric(players["minutes"], errors="coerce").fillna(0.0)
                players["sigma_m"] = 0.0  # oracle minutes for this diagnostic
                players["player_id"] = players["element"].astype(str)
                if not rates.empty:
                    players = players.merge(rates, on="element", how="left")
                if not priors.empty:
                    players = players.merge(
                        priors.drop(columns=["position"], errors="ignore"),
                        on="element",
                        how="left",
                    )
                for c, default in (
                    ("rG", 0.0), ("drG", 0.0), ("rA", 0.0), ("drA", 0.0),
                    ("bonus_mu", 0.0), ("bonus_se", 0.0),
                    ("card_mu", 0.0), ("card_se", 0.0),
                    ("saves_mu", 0.0), ("saves_se", 0.0),
                    ("defcon_mu", 0.0), ("defcon_sig", numagent_xp.SIG_FLOOR),
                    ("defcon_n", 0),
                ):
                    if c not in players.columns:
                        players[c] = default
                    else:
                        players[c] = pd.to_numeric(players[c], errors="coerce").fillna(default)
                scored = numagent_xp.score_team_side(
                    players,
                    xg=mom["xg"],
                    xcs=mom["xcs"],
                    xgc=mom["xgc"],
                    alpha_g=alpha_g,
                    alpha_a=alpha_a,
                )
                scored["element"] = players["element"].to_numpy()
                scored["total_points"] = players["total_points"].to_numpy(float)
                scored["minutes"] = players["m"].to_numpy(float)
                scored = scored.merge(apps, on="element", how="left")
                scored["n_prior"] = scored["n_prior"].fillna(0)
                keep = scored.loc[
                    (scored["minutes"] > MIN_MINUTES)
                    & (scored["n_prior"] >= MIN_HISTORY)
                ]
                for row in keep.itertuples(index=False):
                    xp = float(row.xp)
                    if not math.isfinite(xp):
                        continue
                    xp_vals.append(xp)
                    y_vals.append(float(row.total_points))
    if not xp_vals:
        return {"label": label, "n": 0}
    y = np.asarray(y_vals, float)
    mu = np.asarray(xp_vals, float)
    log = gaussian_log_score(y, mu, SIGMA)
    return {
        "label": label,
        "n": int(len(y)),
        "mean_log": float(log.mean()),
        "mean_xp": float(mu.mean()),
        "mean_points": float(y.mean()),
        "mae": float(np.abs(y - mu).mean()),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "bias": float((mu - y).mean()),
        "scorer": SCORER,
    }


def market_score_xp_v2_control(season: str = TARGET["season"]) -> dict[str, Any]:
    """Published closed ``score_xp_v2`` on the same eligible window."""
    path = PROCESSED / "score_xp_v2_closed.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["season"].astype(str) == str(season)].copy()
    frame["gw"] = pd.to_numeric(frame["gw"], errors="coerce")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0)
    frame["total_points"] = pd.to_numeric(frame["total_points"], errors="coerce")
    frame[SCORER] = pd.to_numeric(frame[SCORER], errors="coerce")
    elig = frame.loc[
        (frame["gw"] >= GW_START)
        & (frame["gw"] <= GW_END)
        & (frame["minutes"] > MIN_MINUTES)
        & frame[SCORER].notna()
        & frame["total_points"].notna()
    ]
    y = elig["total_points"].to_numpy(float)
    mu = elig[SCORER].to_numpy(float)
    log = gaussian_log_score(y, mu, SIGMA)
    return {
        "label": "market_fd_score_xp_v2",
        "n": int(len(elig)),
        "mean_log": float(log.mean()),
        "mean_xp": float(mu.mean()),
        "mean_points": float(y.mean()),
        "mae": float(np.abs(y - mu).mean()),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "bias": float((mu - y).mean()),
        "scorer": SCORER,
    }


def run() -> dict[str, Any]:
    e0 = TARGET["e0"]
    ols = fit_rank_ols_excluding(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    map_kw = _map_kwargs(ols, profile)

    oracle_table = _season_table(CACHE / e0)
    oracle = {norm_team(r["team"]): float(r["rank"]) for r in oracle_table}
    prior = prior_season_ranks(set(oracle))
    flat = flat_ranks(set(oracle))

    fixtures = load_football_data(code=TARGET["code"])
    # Ensure goal columns exist with home_norm/away_norm.
    if "home_goals" not in fixtures.columns and "FTHG" in fixtures.columns:
        fixtures = fixtures.rename(columns={"FTHG": "home_goals", "FTAG": "away_goals"})

    club = {
        "oracle": club_season_errors(oracle_table, oracle, ols, profile),
        "prior_rank": club_season_errors(oracle_table, prior, ols, profile),
        "flat": club_season_errors(oracle_table, flat, ols, profile),
    }
    match = {
        "oracle": match_lambda_scores(fixtures, oracle, map_kw),
        "prior_rank": match_lambda_scores(fixtures, prior, map_kw),
        "flat": match_lambda_scores(fixtures, flat, map_kw),
    }
    mc = {
        "oracle": mc_match_calibration(fixtures, oracle, map_kw),
        "prior_rank": mc_match_calibration(fixtures, prior, map_kw),
        "flat": mc_match_calibration(fixtures, flat, map_kw),
    }

    merged = load_merged(TARGET["season"])
    # Expanding α from finished weeks only is overkill here; use locked pooled α.
    alpha_g, alpha_a, _, _ = numagent_xp.load_alphas()
    market = market_score_xp_v2_control(TARGET["season"])
    player = {
        "oracle": player_cascade_scores(
            merged, oracle, map_kw, label="oracle", alpha_g=alpha_g, alpha_a=alpha_a
        ),
        "prior_rank": player_cascade_scores(
            merged, prior, map_kw, label="prior_rank", alpha_g=alpha_g, alpha_a=alpha_a
        ),
        "flat": player_cascade_scores(
            merged, flat, map_kw, label="flat", alpha_g=alpha_g, alpha_a=alpha_a
        ),
        "market": market,
    }

    # Viability: oracle should beat flat and prior on match MSE and player log.
    viable = (
        match["oracle"]["mse"] < match["flat"]["mse"]
        and match["oracle"]["mse"] <= match["prior_rank"]["mse"]
        and player["oracle"]["mean_log"] > player["flat"]["mean_log"]
        and player["oracle"]["mean_log"] >= player["prior_rank"]["mean_log"]
    )
    vs_market = player["oracle"]["mean_log"] - player["market"]["mean_log"]

    payload = {
        "kind": "oracle_rank_cascade_diagnostic",
        "target": TARGET,
        "map_fit": {
            "exclude": ols["exclude"],
            "files": ols["files"],
            "n_rows": ols["n_rows"],
            "ols_gf": list(ols["ols_gf"]),
            "ols_ga": list(ols["ols_ga"]),
            "mae_gf": ols["mae_gf"],
            "mae_ga": ols["mae_ga"],
            "league_avg_gf": map_kw["league_gf"],
        },
        "oracle_ranks": oracle,
        "prior_ranks": prior,
        "promoted_assigned": {
            c: prior[c] for c in sorted(set(oracle) - set(season_final_ranks(PRIOR_E0)))
        },
        "club_season": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in club.items()},
        "match_lambda": match,
        "mc_calibration": mc,
        "player_gw": player,
        "viable_vs_controls": viable,
        "oracle_minus_market_log": vs_market,
        "scorer": SCORER,
        "notes": (
            "Oracle uses final 2025/26 ranks with prior-only OLS. "
            "Player xP = Opta λ → Dixon–Coles M → score_team_side (score_xp_v2). "
            "Viable if oracle beats flat and is at least as good as prior-season ranks "
            "on match MSE and eligible player-GW log score. Market FD score_xp_v2 is a "
            "reference, not a fail bar for this ceiling test."
        ),
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "oracle_rank_cascade_2025_26.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    # Detail club rows
    pd.DataFrame(club["oracle"]["rows"]).to_csv(
        PROCESSED / "oracle_rank_cascade_club_errors.csv", index=False
    )

    lines = _report_lines(payload)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "oracle_rank_cascade_2025_26.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
    return payload


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def _report_lines(payload: dict[str, Any]) -> list[str]:
    m = payload["map_fit"]
    match = payload["match_lambda"]
    player = payload["player_gw"]
    club = payload["club_season"]
    mc = payload["mc_calibration"]
    lines = [
        "# Oracle-rank cascade diagnostic — 2025/26",
        "",
        "Retrospective ceiling test. Final 2025/26 table ranks are treated as "
        "**predicted** ranks. Rank → GF/GA OLS is fit on **2022/23–2024/25 only**. "
        "Match λ from those season rates; player xP = Dixon–Coles `M` → "
        f"`score_team_side` (`{SCORER}`). Not a live method. 2026/27 unused.",
        "",
        "## Map fit (prior seasons)",
        "",
        f"- Files: {', '.join(m['files'])} (exclude `{m['exclude']}`)",
        f"- GF ≈ {m['ols_gf'][0]:.2f} + ({m['ols_gf'][1]:.3f})·rank  (MAE {m['mae_gf']:.2f})",
        f"- GA ≈ {m['ols_ga'][0]:.2f} + ({m['ols_ga'][1]:.3f})·rank  (MAE {m['mae_ga']:.2f})",
        f"- League avg GF (prior profile): {m['league_avg_gf']:.2f}",
        "",
        f"Promoted clubs (prior-rank control): {payload['promoted_assigned']}",
        "",
        "## Club-season GF/GA (predicted from ranks vs realised)",
        "",
        "| variant | MAE GF | MAE GA | corr GF | corr GA |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("oracle", "prior_rank", "flat"):
        c = club[key]
        lines.append(
            f"| {key} | {c['mae_gf']:.2f} | {c['mae_ga']:.2f} | "
            f"{c['corr_gf']:.3f} | {c['corr_ga']:.3f} |"
        )
    lines += [
        "",
        "## Match λ vs actual goals",
        "",
        "| variant | n fixtures | MSE | MAE | RMSE | band coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("oracle", "prior_rank", "flat"):
        r = match[key]
        lines.append(
            f"| {key} | {r['n_fixtures']} | {r['mse']:.3f} | {r['mae']:.3f} | "
            f"{r['rmse']:.3f} | {r['band_coverage']:.3f} |"
        )
    lines += [
        "",
        f"MC strands (N={mc['oracle']['n_strands']}): oracle MAE(strand mean)="
        f"{mc['oracle']['mae_strand_mean']:.3f} vs MAE(centreline λ)="
        f"{mc['oracle']['mae_centreline_lam']:.3f}.",
        "",
        f"## Eligible player-GW cascade `{SCORER}` (GW5–38, n_prior≥3, minutes>60)",
        "",
        "| variant | n | mean log | MAE | bias | mean xP |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("oracle", "prior_rank", "flat", "market"):
        r = player[key]
        lines.append(
            f"| {r['label']} | {r['n']} | {r['mean_log']:.5f} | {r['mae']:.3f} | "
            f"{r['bias']:+.3f} | {r['mean_xp']:.3f} |"
        )
    lines += [
        "",
        "## Viability",
        "",
        f"**{'PASS' if payload['viable_vs_controls'] else 'FAIL'}** vs flat / prior-rank controls "
        f"(match MSE + player log).",
        f"Oracle − market `{SCORER}` log-score: {payload['oracle_minus_market_log']:+.5f} "
        "(reference only; market uses FD AH+OU pots).",
        "",
        "## Outputs",
        "",
        "- `data/processed/oracle_rank_cascade_2025_26.json`",
        "- `data/processed/oracle_rank_cascade_club_errors.csv`",
        "",
        "Reproduce: `python -m src.eval.oracle_rank_cascade`.",
        "",
    ]
    return lines


if __name__ == "__main__":
    out = run()
    summary = {
        "viable_vs_controls": out["viable_vs_controls"],
        "match_mse": {k: out["match_lambda"][k]["mse"] for k in ("oracle", "prior_rank", "flat")},
        "player_log": {k: out["player_gw"][k]["mean_log"] for k in ("oracle", "prior_rank", "flat", "market")},
        "oracle_minus_market_log": out["oracle_minus_market_log"],
    }
    print(json.dumps(summary, indent=2))
