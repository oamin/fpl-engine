"""Team-GW Dixon–Coles CS / goals calibration (not FPL XI).

Compares opening FD (AH+OU → λ, ρ → xCS/xG) and Opta centreline λ → DC
moments against realised team sheets and goals.

Primary question: is model P(CS) systematically low (as the multi-seed XI
screen suggested), or was that window/XI specific?

  PYTHONPATH=. python3 -m src.eval.dixon_coles_team_cs_calib
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import SEASON_E0
from src.markets.cascade_score_v2 import side_moments_from_lams
from src.markets.numagent_xp_pilot import load_merged
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    match_prediction,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.strategy.layer4_bookie_path import _fd_index
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "dixon_coles_team_cs_calib.md"
STATE = ROOT / "REPORT_STATE.json"
OUT_DEFAULT = Path("/opt/cursor/artifacts/dixon_coles_team_cs_calib.json")


def _now() -> float:
    return time.time()


def _team_gw_sides(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per (GW, fixture, team) with gf/ga/cs and join keys."""
    g = (
        merged.groupby(["GW", "fixture", "team", "was_home"], as_index=False)
        .agg(
            team_h_score=("team_h_score", "first"),
            team_a_score=("team_a_score", "first"),
            date_iso=("date_iso", "first"),
            team_n=("team_n", "first"),
        )
    )
    g["was_home"] = g["was_home"].astype(bool)
    g["gf"] = np.where(
        g["was_home"],
        pd.to_numeric(g["team_h_score"], errors="coerce"),
        pd.to_numeric(g["team_a_score"], errors="coerce"),
    ).astype(float)
    g["ga"] = np.where(
        g["was_home"],
        pd.to_numeric(g["team_a_score"], errors="coerce"),
        pd.to_numeric(g["team_h_score"], errors="coerce"),
    ).astype(float)
    g = g.dropna(subset=["gf", "ga"]).copy()
    g["cs"] = (g["ga"] <= 0).astype(float)
    g["team_n"] = g["team_n"].fillna(g["team"].map(lambda t: norm_team(str(t))))
    g["team_n"] = g["team_n"].map(lambda t: norm_team(str(t)))
    g["date_iso"] = g["date_iso"].astype(str).str.slice(0, 10)
    g["GW"] = pd.to_numeric(g["GW"], errors="coerce").astype(int)
    return g


def _attach_fd(sides: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    f = fd.copy()
    f["team_n"] = f["team_n"].map(lambda t: norm_team(str(t)))
    f["date_iso"] = f["date_iso"].astype(str).str.slice(0, 10)
    # λ conceded = opponent attack
    f["lam_concede"] = np.where(
        f["side"].astype(str).str.lower().eq("home"),
        pd.to_numeric(f["lam_a"], errors="coerce"),
        pd.to_numeric(f["lam_h"], errors="coerce"),
    )
    f["lam_for"] = np.where(
        f["side"].astype(str).str.lower().eq("home"),
        pd.to_numeric(f["lam_h"], errors="coerce"),
        pd.to_numeric(f["lam_a"], errors="coerce"),
    )
    keep = [
        "team_n",
        "date_iso",
        "xg",
        "xcs",
        "xgc",
        "lam_for",
        "lam_concede",
        "rho",
        "lam_h",
        "lam_a",
    ]
    out = sides.merge(f[keep], on=["team_n", "date_iso"], how="inner")
    return out


def _attach_opta(
    sides: pd.DataFrame,
    *,
    e0: str,
) -> pd.DataFrame:
    """Oracle final ranks → centreline λ → DC xCS (leaky ranks; diagnostic)."""
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    # Need home/away names per fixture
    rows: list[dict[str, Any]] = []
    # rebuild fixture pairs
    fx = (
        sides.groupby(["GW", "fixture"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "home": str(g.loc[g["was_home"], "team"].iloc[0])
                    if g["was_home"].any()
                    else None,
                    "away": str(g.loc[~g["was_home"], "team"].iloc[0])
                    if (~g["was_home"]).any()
                    else None,
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    # pandas apply quirk — rebuild simply
    pairs: dict[tuple[int, int], tuple[str, str]] = {}
    for (gw, fid), g in sides.groupby(["GW", "fixture"]):
        homes = g.loc[g["was_home"], "team"]
        aways = g.loc[~g["was_home"], "team"]
        if homes.empty or aways.empty:
            continue
        pairs[(int(gw), int(fid))] = (str(homes.iloc[0]), str(aways.iloc[0]))

    for r in sides.itertuples(index=False):
        key = (int(r.GW), int(r.fixture))
        if key not in pairs:
            continue
        home, away = pairs[key]
        hn, an = norm_team(home), norm_team(away)
        if hn not in ranks or an not in ranks:
            continue
        try:
            pred = match_prediction(hn, an, ranks=ranks, profile=profile)
        except Exception:
            continue
        if bool(r.was_home):
            mom = side_moments_from_lams(pred["lam_home"], pred["lam_away"], rho=0.0)
            lam_for = float(pred["lam_home"])
            lam_conced = float(pred["lam_away"])
        else:
            mom = side_moments_from_lams(pred["lam_away"], pred["lam_home"], rho=0.0)
            lam_for = float(pred["lam_away"])
            lam_conced = float(pred["lam_home"])
        rows.append(
            {
                "GW": int(r.GW),
                "fixture": int(r.fixture),
                "team_n": str(r.team_n),
                "opta_xcs": float(mom["xcs"]),
                "opta_xg": float(mom["xg"]),
                "opta_xgc": float(mom["xgc"]),
                "opta_lam_for": lam_for,
                "opta_lam_concede": lam_conced,
                "opta_p_cs_pois": float(math.exp(-lam_conced)),
            }
        )
    if not rows:
        return sides.iloc[0:0].copy()
    opt = pd.DataFrame(rows)
    return sides.merge(opt, on=["GW", "fixture", "team_n"], how="inner")


def _summarise_cs(df: pd.DataFrame, p_col: str, *, label: str) -> dict[str, Any]:
    p = pd.to_numeric(df[p_col], errors="coerce")
    y = pd.to_numeric(df["cs"], errors="coerce")
    m = p.notna() & y.notna()
    p, y = p[m], y[m]
    if len(p) == 0:
        return {"label": label, "n": 0}
    resid = p - y
    # reliability bins
    bins = np.linspace(0, 1, 6)
    dig = np.digitize(p.to_numpy(float), bins[1:-1], right=False)
    rel = []
    for b in range(5):
        mask = dig == b
        if mask.sum() == 0:
            continue
        rel.append(
            {
                "bin": int(b),
                "p_lo": float(bins[b]),
                "p_hi": float(bins[b + 1]),
                "n": int(mask.sum()),
                "mean_p": float(p.to_numpy(float)[mask].mean()),
                "mean_y": float(y.to_numpy(float)[mask].mean()),
            }
        )
    return {
        "label": label,
        "n": int(len(p)),
        "mean_p_cs": float(p.mean()),
        "mean_cs": float(y.mean()),
        "bias_p_minus_y": float(resid.mean()),
        "brier": float(((p - y) ** 2).mean()),
        "mae": float(resid.abs().mean()),
        "reliability": rel,
    }


def _summarise_goals(
    df: pd.DataFrame, xg_col: str, lam_col: str | None, *, label: str
) -> dict[str, Any]:
    xg = pd.to_numeric(df[xg_col], errors="coerce")
    gf = pd.to_numeric(df["gf"], errors="coerce")
    m = xg.notna() & gf.notna()
    xg, gf = xg[m], gf[m]
    out: dict[str, Any] = {
        "label": label,
        "n": int(len(xg)),
        "mean_xg": float(xg.mean()) if len(xg) else float("nan"),
        "mean_gf": float(gf.mean()) if len(gf) else float("nan"),
        "bias_xg_minus_gf": float((xg - gf).mean()) if len(xg) else float("nan"),
        "mae_xg": float((xg - gf).abs().mean()) if len(xg) else float("nan"),
    }
    if lam_col and lam_col in df.columns:
        lam = pd.to_numeric(df.loc[m, lam_col], errors="coerce")
        out["mean_lam_for"] = float(lam.mean())
        out["bias_lam_minus_gf"] = float((lam - gf).mean())
        out["mae_lam"] = float((lam - gf).abs().mean())
    return out


def _slice_summary(df: pd.DataFrame, p_col: str, xg_col: str, lam_col: str | None) -> dict[str, Any]:
    return {
        "cs": _summarise_cs(df, p_col, label=p_col),
        "goals": _summarise_goals(df, xg_col, lam_col, label=xg_col),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seasons",
        type=str,
        default=",".join(SEASON_E0.keys()),
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in str(args.seasons).split(",") if s.strip()]

    t0 = _now()
    by_season: dict[str, Any] = {}
    pooled_fd: list[pd.DataFrame] = []
    pooled_opta: list[pd.DataFrame] = []

    for season in seasons:
        print(f"[dc-cs] season={season}", flush=True)
        e0 = SEASON_E0[season]
        merged = load_merged(season)
        sides = _team_gw_sides(merged)
        fd = _fd_index(season)
        fd_sides = _attach_fd(sides, fd)
        opta_sides = _attach_opta(sides, e0=e0)
        print(
            f"  sides={len(sides)} fd_join={len(fd_sides)} opta_join={len(opta_sides)}",
            flush=True,
        )
        fd_sides = fd_sides.copy()
        fd_sides["season"] = season
        opta_sides = opta_sides.copy()
        opta_sides["season"] = season
        pooled_fd.append(fd_sides)
        pooled_opta.append(opta_sides)

        # GW8–12 window (the XI screen)
        w = fd_sides.loc[fd_sides["GW"].between(8, 12)]
        by_season[season] = {
            "n_fd": int(len(fd_sides)),
            "n_opta": int(len(opta_sides)),
            "fd_all": _slice_summary(fd_sides, "xcs", "xg", "lam_for"),
            "fd_gw8_12": _slice_summary(w, "xcs", "xg", "lam_for")
            if len(w)
            else {},
            "opta_all": _slice_summary(opta_sides, "opta_xcs", "opta_xg", "opta_lam_for")
            if len(opta_sides)
            else {},
            "opta_gw8_12": _slice_summary(
                opta_sides.loc[opta_sides["GW"].between(8, 12)],
                "opta_xcs",
                "opta_xg",
                "opta_lam_for",
            )
            if len(opta_sides)
            else {},
        }

    fd_all = pd.concat(pooled_fd, ignore_index=True) if pooled_fd else pd.DataFrame()
    opta_all = pd.concat(pooled_opta, ignore_index=True) if pooled_opta else pd.DataFrame()

    summary = {
        "fd_pooled": _slice_summary(fd_all, "xcs", "xg", "lam_for"),
        "fd_pooled_gw8_12": _slice_summary(
            fd_all.loc[fd_all["GW"].between(8, 12)], "xcs", "xg", "lam_for"
        ),
        "opta_pooled": _slice_summary(opta_all, "opta_xcs", "opta_xg", "opta_lam_for"),
        "opta_pooled_gw8_12": _slice_summary(
            opta_all.loc[opta_all["GW"].between(8, 12)],
            "opta_xcs",
            "opta_xg",
            "opta_lam_for",
        ),
        "by_season": by_season,
    }

    # Also Poisson e^{-λ} CS from FD λ_concede
    if len(fd_all):
        fd_all = fd_all.copy()
        fd_all["p_cs_pois"] = np.exp(
            -pd.to_numeric(fd_all["lam_concede"], errors="coerce").clip(lower=0)
        )
        summary["fd_pooled_pois_cs"] = _summarise_cs(fd_all, "p_cs_pois", label="pois")
        summary["fd_gw8_12_pois_cs"] = _summarise_cs(
            fd_all.loc[fd_all["GW"].between(8, 12)], "p_cs_pois", label="pois"
        )

    state = {
        "task": "dixon_coles_team_cs_calib",
        "seasons": seasons,
        "elapsed_s": _now() - t0,
        "note": (
            "Team-GW only. FD = opening AH+OU Dixon–Coles moments. "
            "Opta = oracle final ranks → centreline λ → DC moments (leaky ranks)."
        ),
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    def _cs_line(d: dict[str, Any]) -> str:
        if not d or not d.get("n"):
            return "—"
        return (
            f"n={d['n']}  mean p={d['mean_p_cs']:.3f}  actual={d['mean_cs']:.3f}  "
            f"bias(p−y)={d['bias_p_minus_y']:+.3f}  brier={d['brier']:.3f}"
        )

    def _g_line(d: dict[str, Any]) -> str:
        if not d or not d.get("n"):
            return "—"
        return (
            f"n={d['n']}  mean xG={d['mean_xg']:.3f}  GF={d['mean_gf']:.3f}  "
            f"bias={d['bias_xg_minus_gf']:+.3f}  MAE={d['mae_xg']:.3f}"
        )

    lines = [
        "# Team-GW Dixon–Coles CS / goals calibration",
        "",
        f"Seasons: {', '.join(seasons)} · wall {state['elapsed_s']:.1f}s",
        "",
        "## What this is",
        "",
        "- Unit = **team-GW side** (not FPL XI).",
        "- **FD**: football-data opening AH+OU → Dixon–Coles `(λ_h, λ_a, ρ)` → "
        "`xcs` / `xg` (same moments as L3/L4).",
        "- **Opta**: oracle end-of-season ranks → centreline λ → DC moments "
        "(leaky on ranks; checks whether Opta λ→CS is also low).",
        "- Actual CS = `GA == 0`.",
        "",
        "## Pooled FD (all seasons)",
        "",
        f"- CS (DC xcs): {_cs_line(summary['fd_pooled']['cs'])}",
        f"- CS (Poisson e^{{-λ_concede}}): {_cs_line(summary.get('fd_pooled_pois_cs') or {})}",
        f"- Goals (xG): {_g_line(summary['fd_pooled']['goals'])}",
        "",
        "### FD GW8–12 only (XI-screen window)",
        "",
        f"- CS (DC xcs): {_cs_line(summary['fd_pooled_gw8_12']['cs'])}",
        f"- CS (Poisson): {_cs_line(summary.get('fd_gw8_12_pois_cs') or {})}",
        f"- Goals: {_g_line(summary['fd_pooled_gw8_12']['goals'])}",
        "",
        "## Pooled Opta centreline",
        "",
        f"- CS: {_cs_line(summary['opta_pooled']['cs'])}",
        f"- Goals: {_g_line(summary['opta_pooled']['goals'])}",
        "",
        "### Opta GW8–12",
        "",
        f"- CS: {_cs_line(summary['opta_pooled_gw8_12']['cs'])}",
        f"- Goals: {_g_line(summary['opta_pooled_gw8_12']['goals'])}",
        "",
        "## By season (FD all GWs)",
        "",
        "| season | n | mean p_CS | actual CS | bias p−y | mean xG | GF | bias xG−GF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for season in seasons:
        cs = by_season[season]["fd_all"]["cs"]
        gl = by_season[season]["fd_all"]["goals"]
        lines.append(
            f"| {season} | {cs['n']} | {cs['mean_p_cs']:.3f} | {cs['mean_cs']:.3f} | "
            f"{cs['bias_p_minus_y']:+.3f} | {gl['mean_xg']:.3f} | {gl['mean_gf']:.3f} | "
            f"{gl['bias_xg_minus_gf']:+.3f} |"
        )
    lines += [
        "",
        "## FD reliability (pooled)",
        "",
        "| p bin | n | mean p | actual CS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for b in summary["fd_pooled"]["cs"].get("reliability") or []:
        lines.append(
            f"| {b['p_lo']:.1f}–{b['p_hi']:.1f} | {b['n']} | {b['mean_p']:.3f} | "
            f"{b['mean_y']:.3f} |"
        )
    # Verdict
    fd_bias = summary["fd_pooled"]["cs"]["bias_p_minus_y"]
    fd_w = summary["fd_pooled_gw8_12"]["cs"]["bias_p_minus_y"]
    g_bias = summary["fd_pooled"]["goals"]["bias_xg_minus_gf"]
    lines += [
        "",
        "## Verdict",
        "",
        (
            f"FD DC xCS bias (p−actual) pooled **{fd_bias:+.3f}**; "
            f"GW8–12 **{fd_w:+.3f}**. "
            f"Goals xG−GF pooled **{g_bias:+.3f}**. "
            + (
                "CS is systematically **under**-predicted at team-GW → "
                "consistent with a DC / λ→CS calibration issue (not only XI luck)."
                if fd_bias < -0.02
                else (
                    "CS bias is small at team-GW → the XI CS gap was mostly "
                    "selection/window, not a broken DC prior."
                    if abs(fd_bias) <= 0.02
                    else "CS is systematically **over**-predicted at team-GW."
                )
            )
        ),
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.dixon_coles_team_cs_calib",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "dixon_coles_team_cs_calib",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Team-GW FD/Opta CS+goals calibration vs actual sheets",
                "verdict": {
                    "fd_cs_bias": fd_bias,
                    "fd_cs_bias_gw8_12": fd_w,
                    "fd_xg_bias": g_bias,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(state["summary"]["fd_pooled"], indent=2, default=str)[:1500])
    print("wrote", args.out, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
