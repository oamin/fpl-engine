"""Oracle-rank cascade diagnostic on a closed season (default 2025/26).

Uses that season's **final table** as predicted ranks (oracle), maps rank →
season GF/GA with OLS fit on **prior** closed seasons only, then scores the
Opta pot cascade (Dixon–Coles λ → frozen shares × ``xp_on_pot``) against
realised match goals and player points.

Controls: prior-season ranks (promoted clubs → 18/19/20), flat mid-table
rank 10.5, and the market Shin λ already on the joined frame.

This is a retrospective ceiling, not a live method and not a promote gate
for ``score_xp``. 2026/27 is not used.
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
from src.ingest.fpl_odds import join_players_to_fixtures, load_football_data, load_player_logs
from src.models.forecast_xp import xp_on_pot
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
from src.models.xp_engine import (
    DEFCON_THRESH,
    MIN_MINUTES,
    add_market_pots,
    add_player_priors,
    compute_xp,
)
from src.teams import norm_team

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
MIN_XMI = 45.0
SIGMA = 3.0
N_MC = 40
MC_SEED = 0


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gw"] = pd.to_numeric(out["gw"], errors="coerce")
    out = out.dropna(subset=["gw", "position"]).copy()
    out["gw"] = out["gw"].astype(int)
    minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
    out["minutes"] = minutes
    out["total_points"] = pd.to_numeric(out["total_points"], errors="coerce").fillna(0.0)
    out["xG"] = pd.to_numeric(out["xG"], errors="coerce").fillna(0.0)
    out["xA"] = pd.to_numeric(out["xA"], errors="coerce").fillna(0.0)
    out["goals"] = pd.to_numeric(out.get("goals", 0.0), errors="coerce").fillna(0.0)
    out["value"] = pd.to_numeric(out.get("value"), errors="coerce")
    raw = pd.to_numeric(out.get("defcon", 0.0), errors="coerce").fillna(0.0)
    thr = out["position"].map(DEFCON_THRESH)
    out["defcon_hit"] = (
        out["position"].isin(list(DEFCON_THRESH))
        & (minutes >= MIN_MINUTES)
        & (raw >= thr.fillna(999))
    ).astype(float)
    if "p_not_lose" not in out.columns:
        out["p_not_lose"] = np.nan
    out["p_not_lose"] = pd.to_numeric(out["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        out["p_draw"], errors="coerce"
    )
    if "attack_strength" not in out.columns:
        out["attack_strength"] = np.nan
    if "defend_threat" not in out.columns:
        out["defend_threat"] = np.nan
    # Opponent norms for pot lookup.
    out["opp_norm"] = np.where(
        out["is_home"].astype(bool),
        out.get("away_norm", pd.Series(index=out.index, dtype=object)),
        out.get("home_norm", pd.Series(index=out.index, dtype=object)),
    )
    out["opp_norm"] = out["opp_norm"].map(lambda x: norm_team(x) if pd.notna(x) else "")
    out["team_norm"] = out["team_norm"].map(lambda x: norm_team(x) if pd.notna(x) else "")
    return out


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
    scored: pd.DataFrame,
    ranks: dict[str, float],
    map_kw: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Eligible player-GW log score of cascade xP vs realised points."""
    window = scored.loc[
        (scored["gw"] >= GW_START)
        & (scored["gw"] <= GW_END)
        & (scored["n_prior"] >= MIN_HISTORY)
        & (scored["xmi"] >= MIN_XMI)
    ].copy()
    xp_vals = []
    y_vals = []
    for row in window.itertuples(index=False):
        club = str(row.team_norm)
        opp = str(row.opp_norm)
        if club not in ranks or opp not in ranks or not opp:
            continue
        is_home = bool(row.is_home)
        home, away = (club, opp) if is_home else (opp, club)
        try:
            pred = match_prediction(home, away, ranks=ranks, **map_kw)
        except KeyError:
            continue
        pot = pred["pot_home"] if is_home else pred["pot_away"]
        xp = xp_on_pot(
            position=str(row.position),
            xmi=float(row.xmi) if math.isfinite(float(row.xmi)) else 0.0,
            share_xg=float(row.share_xG),
            share_xa=float(row.share_xA),
            exp_defcon_hit=float(row.exp_defcon_hit) if math.isfinite(float(row.exp_defcon_hit)) else 0.0,
            fwd_goal_scale=float(getattr(row, "fwd_goal_scale", 1.0) or 1.0),
            pot=pot,
        )
        if not math.isfinite(xp):
            continue
        xp_vals.append(xp)
        y_vals.append(float(row.total_points))
    if not xp_vals:
        return {"label": label, "n": 0}
    y = np.asarray(y_vals, float)
    mu = np.asarray(xp_vals, float)
    log = gaussian_log_score(y, mu, SIGMA)
    # Market / published score_xp on the same rows for paired delta later.
    return {
        "label": label,
        "n": int(len(y)),
        "mean_log": float(log.mean()),
        "mean_xp": float(mu.mean()),
        "mean_points": float(y.mean()),
        "mae": float(np.abs(y - mu).mean()),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "bias": float((mu - y).mean()),
    }


def build_scored_season(season: str, code: str) -> pd.DataFrame:
    players = load_player_logs(season=season)
    fixtures = load_football_data(code=code)
    joined, _, _ = join_players_to_fixtures(players, fixtures, retain_sheet_rows=True)
    frame = _prepare(joined)
    frame = add_market_pots(frame)
    frame = add_player_priors(frame)
    frame = compute_xp(frame)
    return frame


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

    scored = build_scored_season(TARGET["season"], TARGET["code"])
    # Market control: published compute_xp on the same eligible window.
    elig = scored.loc[
        (scored["gw"] >= GW_START)
        & (scored["gw"] <= GW_END)
        & (scored["n_prior"] >= MIN_HISTORY)
        & (scored["xmi"] >= MIN_XMI)
        & scored["xp"].notna()
        & scored["total_points"].notna()
    ]
    y = elig["total_points"].to_numpy(float)
    market_log = gaussian_log_score(y, elig["xp"].to_numpy(float), SIGMA)
    market = {
        "label": "market_shin_score_xp",
        "n": int(len(elig)),
        "mean_log": float(market_log.mean()),
        "mean_xp": float(elig["xp"].mean()),
        "mean_points": float(y.mean()),
        "mae": float(np.abs(y - elig["xp"].to_numpy(float)).mean()),
        "rmse": float(np.sqrt(np.mean((y - elig["xp"].to_numpy(float)) ** 2))),
        "bias": float((elig["xp"].to_numpy(float) - y).mean()),
    }

    player = {
        "oracle": player_cascade_scores(scored, oracle, map_kw, label="oracle"),
        "prior_rank": player_cascade_scores(scored, prior, map_kw, label="prior_rank"),
        "flat": player_cascade_scores(scored, flat, map_kw, label="flat"),
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
        "notes": (
            "Oracle uses final 2025/26 ranks with prior-only OLS. "
            "Viable if oracle beats flat and is at least as good as prior-season ranks "
            "on match MSE and eligible player-GW log score. Market Shin score_xp is a "
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
        "Match pots use Dixon–Coles from those season rates; player xP uses deadline "
        "shares × `xp_on_pot` (cascade path). Not a live method. 2026/27 unused.",
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
        "## Eligible player-GW cascade xP (GW5–38, n_prior≥3, xmi≥45)",
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
        f"Oracle − market log-score: {payload['oracle_minus_market_log']:+.5f} "
        "(reference only; market uses fixture odds).",
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
