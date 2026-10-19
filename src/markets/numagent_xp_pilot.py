"""2025/26 Numagent 2.0 walk-forward pilot vs score_xp.

Pilot only (not gated): FD opening AH+OU moments → as-of priors →
``score_team_side`` on players with minutes > 60. Primary metric is
Gaussian log score on the point mean (σ=3) vs published ``score_xp``.
``xp_se`` is secondary calibration only.

  python -m src.markets.numagent_xp_pilot
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.gates import gaussian_log_score
from src.markets import fpl_score_rates, numagent_xp
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "data" / "plots"

SEASON = "2025_26"
SIGMA = 3.0
MIN_MINUTES = 60.0
HISTORY_MIN_GW = 2  # first scored GW (priors from GW < t)
GW_EVAL_START = 5  # protocol eval window start


def _asof_alphas(history: pd.DataFrame) -> tuple[float, float]:
    """Expanding α_G, α_A from finished fixture-sides only (no season leak)."""
    if history.empty:
        return 0.96, 0.90
    rows: list[dict[str, Any]] = []
    for (_, _), g in history.groupby(["fixture", "was_home"], sort=False):
        wh_b = bool(g["was_home"].iloc[0])
        pg = float(pd.to_numeric(g["goals_scored"], errors="coerce").fillna(0).sum())
        pa = float(pd.to_numeric(g["assists"], errors="coerce").fillna(0).sum())
        hs = g["team_h_score"].iloc[0]
        aws = g["team_a_score"].iloc[0]
        if pd.isna(hs) or pd.isna(aws):
            continue
        mg = float(hs if wh_b else aws)
        rows.append(
            {
                "fixture_id": g["fixture"].iloc[0],
                "player_goals": pg,
                "assists": pa,
                "match_goals": mg,
            }
        )
    if not rows:
        return 0.96, 0.90
    clean = fpl_score_rates.clean_sides(pd.DataFrame(rows))
    mg = float(clean["match_goals"].sum())
    if mg <= 0:
        return 0.96, 0.90
    return float(clean["player_goals"].sum() / mg), float(clean["assists"].sum() / mg)


def _season_stem(season: str) -> str:
    """Accept ``2025_26`` or ``2025-26``."""
    return season.replace("-", "_")


def load_merged(season: str = SEASON) -> pd.DataFrame:
    stem = _season_stem(season)
    path = CACHE / f"merged_gw_{stem}.csv"
    cols = [
        "name",
        "element",
        "position",
        "team",
        "GW",
        "fixture",
        "kickoff_time",
        "was_home",
        "minutes",
        "total_points",
        "goals_scored",
        "assists",
        "bonus",
        "yellow_cards",
        "red_cards",
        "team_h_score",
        "team_a_score",
    ]
    header = pd.read_csv(path, nrows=0).columns
    if "defensive_contribution" in header:
        cols = cols + ["defensive_contribution"]
    if "saves" in header:
        cols = cols + ["saves"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    if "defensive_contribution" not in df.columns:
        df["defensive_contribution"] = 0.0
    if "saves" not in df.columns:
        df["saves"] = 0.0
    else:
        df["saves"] = pd.to_numeric(df["saves"], errors="coerce").fillna(0.0)
    df["element"] = df["element"].astype(int)
    df["GW"] = pd.to_numeric(df["GW"], errors="coerce").astype(int)
    df["fixture"] = pd.to_numeric(df["fixture"], errors="coerce").astype(int)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce").fillna(0.0)
    df["team_n"] = df["team"].map(norm_team)
    df["date_iso"] = (
        pd.to_datetime(df["kickoff_time"], utc=True).dt.tz_convert(None).dt.date
    )
    # stem ``2022_23`` → label ``2022-23``
    parts = stem.split("_")
    df["season"] = f"{parts[0]}-{parts[1]}" if len(parts) == 2 else season
    return df


def load_fd_moments(season: str = SEASON) -> pd.DataFrame:
    stem = _season_stem(season)
    fd = pd.read_csv(PROCESSED / f"scoreline_fd_moments_{stem}.csv")
    fd["team_n"] = fd["team"].map(norm_team)
    fd["date_iso"] = pd.to_datetime(fd["date"], dayfirst=True).dt.date
    keep = ["team_n", "date_iso", "xg", "xcs", "xgc", "side", "home", "away"]
    return fd[keep].drop_duplicates(["team_n", "date_iso"], keep="first")


def load_score_xp(season: str | None = None) -> pd.DataFrame:
    """Load score_xp. Prefer closed-season rebuild; fall back to xp_engine for 2025/26."""
    if season is None or _season_stem(season) == "2025_26":
        closed = PROCESSED / f"score_xp_{_season_stem(season or SEASON)}.csv"
        if closed.exists():
            xp = pd.read_csv(
                closed,
                usecols=["element", "gw", "fixture_id", "score_xp", "total_points", "minutes", "n_prior", "xmi"],
            )
            xp["player_id"] = xp["element"].astype(int)
            xp["gw"] = xp["gw"].astype(int)
            xp["fixture_id"] = xp["fixture_id"].astype(str)
            return xp
        xp = pd.read_csv(
            PROCESSED / "xp_engine.csv",
            usecols=["player_id", "gw", "fixture_id", "xp", "total_points", "minutes"],
        )
        xp["player_id"] = xp["player_id"].astype(int)
        xp["gw"] = xp["gw"].astype(int)
        xp["fixture_id"] = xp["fixture_id"].astype(str)
        return xp.rename(columns={"xp": "score_xp", "total_points": "score_xp_points"})

    stem = _season_stem(season)
    xp = pd.read_csv(
        PROCESSED / f"score_xp_{stem}.csv",
        usecols=["element", "gw", "fixture_id", "score_xp", "total_points", "minutes", "n_prior", "xmi"],
    )
    xp["player_id"] = xp["element"].astype(int)
    xp["gw"] = xp["gw"].astype(int)
    xp["fixture_id"] = xp["fixture_id"].astype(str)
    return xp


def _fixture_key(date_iso, home: str, away: str) -> str:
    return f"{date_iso}:{norm_team(home)}:{norm_team(away)}"


def score_gameweek(
    hist: pd.DataFrame,
    week: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
    m0: float | None = None,
    rate_se_floor: float | None = None,
) -> pd.DataFrame:
    """Score one GW: shares over minutes>0; keep minutes>60 for the pilot pool.

    ``hist`` must be same-season GW ``< t`` (no prior-season fill for shares).
    Optional ``m0`` / ``rate_se_floor`` pass through to ``player_rates_from_history``
    (defaults: promoted RATE_M0 / RATE_SE_FLOOR).
    """
    if week.empty:
        return pd.DataFrame()

    numagent_xp.assert_single_season_rate_history(hist)
    rates = numagent_xp.player_rates_from_history(
        hist, id_col="element", m0=m0, rate_se_floor=rate_se_floor
    )
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")

    # Attach FD moments
    w = week.merge(fd, on=["team_n", "date_iso"], how="left")
    w = w[w["xg"].notna()].copy()
    if w.empty:
        return pd.DataFrame()

    # Appearance players for share renormalisation
    apps = w[w["minutes"] > 0].copy()
    if apps.empty:
        return pd.DataFrame()

    if rates.empty:
        apps["rG"] = 0.0
        apps["drG"] = 0.0
        apps["rA"] = 0.0
        apps["drA"] = 0.0
    else:
        apps = apps.merge(rates, on="element", how="left")
        for c in ("rG", "drG", "rA", "drA"):
            apps[c] = apps[c].fillna(0.0)

    if priors.empty:
        for c in (
            "bonus_mu",
            "bonus_se",
            "card_mu",
            "card_se",
            "saves_mu",
            "saves_se",
            "defcon_mu",
            "defcon_sig",
            "defcon_n",
        ):
            apps[c] = 0.0
        apps["defcon_sig"] = numagent_xp.SIG_FLOOR
    else:
        apps = apps.merge(priors.drop(columns=["position"], errors="ignore"), on="element", how="left")
        apps["bonus_mu"] = apps["bonus_mu"].fillna(0.0)
        apps["bonus_se"] = apps["bonus_se"].fillna(0.0)
        apps["card_mu"] = apps["card_mu"].fillna(0.0)
        apps["card_se"] = apps["card_se"].fillna(0.0)
        apps["saves_mu"] = apps["saves_mu"].fillna(0.0)
        apps["saves_se"] = apps["saves_se"].fillna(0.0)
        apps["defcon_mu"] = apps["defcon_mu"].fillna(0.0)
        apps["defcon_sig"] = apps["defcon_sig"].fillna(numagent_xp.SIG_FLOOR)
        apps["defcon_n"] = apps["defcon_n"].fillna(0).astype(int)

    # Pilot: actual minutes, σ_m = 0 (oracle minutes, stack test)
    apps["m"] = apps["minutes"].astype(float)
    apps["sigma_m"] = 0.0
    apps["player_id"] = apps["element"]

    scored_parts: list[pd.DataFrame] = []
    for (_, _), g in apps.groupby(["fixture", "team_n"], sort=False):
        side = numagent_xp.score_team_side(
            g,
            xg=float(g["xg"].iloc[0]),
            xcs=float(g["xcs"].iloc[0]),
            xgc=float(g["xgc"].iloc[0]),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
        )
        scored_parts.append(side)

    if not scored_parts:
        return pd.DataFrame()
    out = pd.concat(scored_parts, ignore_index=True)
    # Pilot evaluation pool
    out = out[out["minutes"] > MIN_MINUTES].copy()
    out["gw"] = int(week["GW"].iloc[0])
    out["alpha_g"] = float(alpha_g)
    out["alpha_a"] = float(alpha_a)
    return out


def run_pilot(season: str = SEASON) -> dict[str, Any]:
    merged = load_merged(season)
    fd = load_fd_moments(season)
    score_xp = load_score_xp(season)

    parts: list[pd.DataFrame] = []
    alpha_rows: list[dict[str, Any]] = []
    gws = sorted(int(g) for g in merged["GW"].unique() if g >= HISTORY_MIN_GW)

    for gw in gws:
        hist = merged[merged["GW"] < gw]
        week = merged[merged["GW"] == gw]
        ag, aa = _asof_alphas(hist)
        alpha_rows.append({"gw": gw, "alpha_g": ag, "alpha_a": aa, "n_hist_rows": len(hist)})
        scored = score_gameweek(hist, week, fd, alpha_g=ag, alpha_a=aa)
        if not scored.empty:
            parts.append(scored)

    if not parts:
        raise RuntimeError("pilot produced no scored rows")

    num = pd.concat(parts, ignore_index=True)
    num["element"] = num["element"].astype(int)
    num["fixture_id"] = [
        _fixture_key(d, h, a)
        for d, h, a in zip(num["date_iso"], num["home"], num["away"], strict=True)
    ]

    joined = num.merge(
        score_xp,
        left_on=["element", "gw", "fixture_id"],
        right_on=["player_id", "gw", "fixture_id"],
        how="inner",
        suffixes=("", "_xp"),
    )
    if joined.empty:
        raise RuntimeError(f"no overlap with score_xp after join ({season})")

    season_label = str(merged["season"].iloc[0])
    joined["season"] = season_label

    y = joined["total_points"].to_numpy(float)
    mu_n = joined["xp"].to_numpy(float)
    mu_s = joined["score_xp"].to_numpy(float)
    joined["log_numagent"] = gaussian_log_score(y, mu_n, SIGMA)
    joined["log_score_xp"] = gaussian_log_score(y, mu_s, SIGMA)
    joined["d_log"] = joined["log_numagent"] - joined["log_score_xp"]
    joined["abs_err_num"] = np.abs(y - mu_n)
    joined["abs_err_sxp"] = np.abs(y - mu_s)
    joined["sq_err_num"] = (y - mu_n) ** 2
    joined["sq_err_sxp"] = (y - mu_s) ** 2

    # Secondary: SE coverage (Numagent only)
    se = joined["xp_se"].to_numpy(float)
    joined["cover_1se"] = (np.abs(y - mu_n) <= se).astype(float)
    joined["cover_2se"] = (np.abs(y - mu_n) <= 2.0 * se).astype(float)

    gw_rows: list[dict[str, Any]] = []
    for gw, g in joined.groupby("gw", sort=True):
        gw_rows.append(
            {
                "gw": int(gw),
                "n": int(len(g)),
                "d_log": float(g["d_log"].mean()),
                "log_numagent": float(g["log_numagent"].mean()),
                "log_score_xp": float(g["log_score_xp"].mean()),
                "mae_num": float(g["abs_err_num"].mean()),
                "mae_sxp": float(g["abs_err_sxp"].mean()),
                "bias_num": float(g["xp"].mean() - g["total_points"].mean()),
                "bias_sxp": float(g["score_xp"].mean() - g["total_points"].mean()),
                "mean_xp": float(g["xp"].mean()),
                "mean_sxp": float(g["score_xp"].mean()),
                "mean_pts": float(g["total_points"].mean()),
                "cover_1se": float(g["cover_1se"].mean()),
                "cover_2se": float(g["cover_2se"].mean()),
                "mean_se": float(g["xp_se"].mean()),
            }
        )
    gw_tbl = pd.DataFrame(gw_rows)

    deltas = gw_tbl["d_log"].to_numpy(float)
    mean_d = float(np.mean(deltas))
    # Single-season GW bootstrap (pilot CI — not a gated four-season interval)
    rng = np.random.default_rng(0)
    boots = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, len(deltas), size=len(deltas))
        boots[i] = float(deltas[idx].mean())
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    # GW 5–38 slice (eval convention)
    late = gw_tbl[gw_tbl["gw"] >= 5]
    late_d = late["d_log"].to_numpy(float)
    late_mean = float(np.mean(late_d)) if len(late_d) else float("nan")
    if len(late_d):
        boots_l = np.empty(1000)
        for i in range(1000):
            idx = rng.integers(0, len(late_d), size=len(late_d))
            boots_l[i] = float(late_d[idx].mean())
        late_lo, late_hi = float(np.quantile(boots_l, 0.025)), float(np.quantile(boots_l, 0.975))
    else:
        late_lo = late_hi = float("nan")

    summary: dict[str, Any] = {
        "season": season_label,
        "pilot": True,
        "gated": False,
        "pool": f"minutes > {int(MIN_MINUTES)}",
        "minutes_policy": "actual minutes, sigma_m=0",
        "odds": "football-data opening AH+OU (BTTS weight 0)",
        "sigma_log_score": SIGMA,
        "n_player_fixture": int(len(joined)),
        "n_gameweeks": int(len(gw_tbl)),
        "gw_range": [int(gw_tbl["gw"].min()), int(gw_tbl["gw"].max())],
        "mean_d_log_all": mean_d,
        "ci95_all": [lo, hi],
        "mean_d_log_gw5_38": late_mean,
        "ci95_gw5_38": [late_lo, late_hi],
        "row_mae_numagent": float(joined["abs_err_num"].mean()),
        "row_mae_score_xp": float(joined["abs_err_sxp"].mean()),
        "row_bias_numagent": float(joined["xp"].mean() - joined["total_points"].mean()),
        "row_bias_score_xp": float(joined["score_xp"].mean() - joined["total_points"].mean()),
        "cover_1se": float(joined["cover_1se"].mean()),
        "cover_2se": float(joined["cover_2se"].mean()),
        "mean_xp_se": float(joined["xp_se"].mean()),
        "positive_delta": "numagent higher log-score than score_xp",
    }

    tmp = merged[merged["minutes"] > MIN_MINUTES].merge(
        fd[["team_n", "date_iso", "xg"]], on=["team_n", "date_iso"], how="left"
    )
    summary["n_pool_rows_no_fd"] = int(tmp["xg"].isna().sum())
    summary["n_pool_rows_with_fd"] = int(tmp["xg"].notna().sum())
    summary["n_joined_score_xp"] = int(len(joined))

    return {
        "rows": joined,
        "gw": gw_tbl,
        "alphas": pd.DataFrame(alpha_rows),
        "summary": summary,
    }


def write_outputs(result: dict[str, Any]) -> dict[str, str]:
    rows: pd.DataFrame = result["rows"]
    gw: pd.DataFrame = result["gw"]
    alphas: pd.DataFrame = result["alphas"]
    summary: dict[str, Any] = result["summary"]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    row_path = PROCESSED / "numagent_xp_pilot_2025_26.csv"
    gw_path = PROCESSED / "numagent_xp_pilot_2025_26_gw.csv"
    alpha_path = PROCESSED / "numagent_xp_pilot_2025_26_alpha.csv"
    json_path = PROCESSED / "numagent_xp_pilot_2025_26_summary.json"
    plot_path = PLOTS / "numagent_xp_pilot_2025_26_dlog.png"
    report_path = REPORTS / "numagent_xp_pilot_2025_26.md"

    keep_cols = [
        c
        for c in [
            "gw",
            "element",
            "name",
            "team",
            "position",
            "fixture",
            "minutes",
            "total_points",
            "xp",
            "xp_se",
            "score_xp",
            "log_numagent",
            "log_score_xp",
            "d_log",
            "xp_appear",
            "xp_goals",
            "xp_assists",
            "xp_cs",
            "xp_gc",
            "xp_defcon",
            "xp_bonus",
            "xp_cards",
            "xg",
            "xcs",
            "xgc",
            "alpha_g",
            "alpha_a",
            "cover_1se",
            "cover_2se",
        ]
        if c in rows.columns
    ]
    rows[keep_cols].to_csv(row_path, index=False)
    gw.to_csv(gw_path, index=False)
    alphas.to_csv(alpha_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot GW Δlog-score
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.plot(gw["gw"], gw["d_log"], marker="o", ms=3.5, lw=1.2, color="#1f4e79")
    ax.fill_between(
        gw["gw"],
        0,
        gw["d_log"],
        where=gw["d_log"] >= 0,
        color="#2a9d8f",
        alpha=0.25,
        interpolate=True,
    )
    ax.fill_between(
        gw["gw"],
        0,
        gw["d_log"],
        where=gw["d_log"] < 0,
        color="#e76f51",
        alpha=0.25,
        interpolate=True,
    )
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Δ log-score (numagent − score_xp)")
    ax.set_title(
        f"Numagent 2.0 pilot · 2025/26 · minutes>{int(MIN_MINUTES)} · "
        f"mean Δ={summary['mean_d_log_all']:+.4f} "
        f"[{summary['ci95_all'][0]:+.4f}, {summary['ci95_all'][1]:+.4f}]"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    s = summary
    lines = [
        "# Numagent 2.0 xP pilot — 2025/26",
        "",
        "Pilot only. Not a gated four-season result. No holdout season. No climb.",
        "",
        "## Protocol",
        "",
        f"- Pool: players with **minutes > {int(MIN_MINUTES)}** and an FD opening AH+OU fit.",
        "- Minutes: **actual** minutes, **σ_m = 0** (stack test, not minutes tagging).",
        "- Odds: football-data opening AH + OU 2.5 (BTTS weight 0).",
        "- Priors / rates / α: walk-forward from GW `< t` only.",
        f"- Primary metric: Gaussian log score, σ = {SIGMA}, on the **point** mean.",
        "- Secondary: Numagent `xp_se` coverage only (not in the log-score).",
        "- Baseline: published `score_xp` from `xp_engine.csv` on the same player-fixtures.",
        "",
        "## Headline",
        "",
        f"- Player-fixtures: **{s['n_joined_score_xp']}** across GWs "
        f"**{s['gw_range'][0]}–{s['gw_range'][1]}**.",
        f"- Mean GW Δlog-score (numagent − score_xp), all GWs: "
        f"**{s['mean_d_log_all']:+.4f}** "
        f"[{s['ci95_all'][0]:+.4f}, {s['ci95_all'][1]:+.4f}] "
        "(single-season GW bootstrap, B=1000, seed 0).",
        f"- Same, GWs 5–38: **{s['mean_d_log_gw5_38']:+.4f}** "
        f"[{s['ci95_gw5_38'][0]:+.4f}, {s['ci95_gw5_38'][1]:+.4f}].",
        f"- Row MAE: numagent **{s['row_mae_numagent']:.3f}** vs score_xp "
        f"**{s['row_mae_score_xp']:.3f}**.",
        f"- Row bias (μ − points): numagent **{s['row_bias_numagent']:+.3f}** vs "
        f"score_xp **{s['row_bias_score_xp']:+.3f}**.",
        "",
        "## Secondary — Numagent SE",
        "",
        f"- Mean `xp_se`: **{s['mean_xp_se']:.3f}**.",
        f"- Coverage within 1 SE: **{s['cover_1se']:.3f}**; within 2 SE: **{s['cover_2se']:.3f}**.",
        "",
        "## Artefacts",
        "",
        f"- `{row_path.relative_to(ROOT)}`",
        f"- `{gw_path.relative_to(ROOT)}`",
        f"- `{alpha_path.relative_to(ROOT)}`",
        f"- `{json_path.relative_to(ROOT)}`",
        f"- `{plot_path.relative_to(ROOT)}`",
        "",
        "Positive Δ means Numagent has the higher log score. This pilot does not "
        "replace published `score_xp`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "rows": str(row_path),
        "gw": str(gw_path),
        "alphas": str(alpha_path),
        "summary": str(json_path),
        "plot": str(plot_path),
        "report": str(report_path),
    }


def main() -> None:
    result = run_pilot()
    paths = write_outputs(result)
    s = result["summary"]
    print(
        f"pilot 2025/26  n={s['n_joined_score_xp']}  "
        f"d_log={s['mean_d_log_all']:+.4f} "
        f"[{s['ci95_all'][0]:+.4f}, {s['ci95_all'][1]:+.4f}]"
    )
    print(
        f"gw5-38  d_log={s['mean_d_log_gw5_38']:+.4f} "
        f"[{s['ci95_gw5_38'][0]:+.4f}, {s['ci95_gw5_38'][1]:+.4f}]"
    )
    print(
        f"MAE num={s['row_mae_numagent']:.3f}  sxp={s['row_mae_score_xp']:.3f}  "
        f"cover1={s['cover_1se']:.3f} cover2={s['cover_2se']:.3f}"
    )
    print(paths["report"])


if __name__ == "__main__":
    main()
