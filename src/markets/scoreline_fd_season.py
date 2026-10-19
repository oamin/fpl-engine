"""Season eval: football-data opening AH+OU → M → xG/xCS/xGC vs actuals."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets.scoreline import fair_pair_from_decimals, fit_scoreline, team_moments

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SEASON_FILES = {
    "2022-23": "E0_2223.csv",
    "2023-24": "E0_2324.csv",
    "2024-25": "E0_2425.csv",
    "2025-26": "E0_2526.csv",
}


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    s = pd.Series(a).corr(pd.Series(b), method=method)
    return float(s) if s == s else float("nan")


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logloss(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": float(mask.sum()),
                "p_mean": float(p[mask].mean()),
                "y_mean": float(y[mask].mean()),
            }
        )
    return out


def fit_season_sides(fd_path: Path) -> pd.DataFrame:
    fd = pd.read_csv(fd_path)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for idx, r in fd.iterrows():
        try:
            ah_line = float(r["AHh"])
            p_ah, _ = fair_pair_from_decimals(float(r["AvgAHH"]), float(r["AvgAHA"]))
            p_over, _ = fair_pair_from_decimals(float(r["Avg>2.5"]), float(r["Avg<2.5"]))
        except (TypeError, ValueError, KeyError):
            skipped += 1
            continue
        if not all(math.isfinite(x) for x in (ah_line, p_ah, p_over)):
            skipped += 1
            continue

        fit = fit_scoreline(
            p_ah=p_ah,
            ah_line=ah_line,
            p_over_25=p_over,
            p_btts=0.5,
            weights=(1.0, 1.0, 0.0),
        )
        mom = team_moments(fit.matrix)
        fthg = int(r["FTHG"])
        ftag = int(r["FTAG"])
        base = {
            "date": str(r["Date"]),
            "home": str(r["HomeTeam"]),
            "away": str(r["AwayTeam"]),
            "ah_line": ah_line,
            "p_ah": p_ah,
            "p_over_25": p_over,
            "lam_h": fit.lam_h,
            "lam_a": fit.lam_a,
            "rho": fit.rho,
            "sse": fit.sse,
            "fit_ok": bool(fit.success),
        }
        rows.append(
            {
                **base,
                "side": "home",
                "team": str(r["HomeTeam"]),
                "opp": str(r["AwayTeam"]),
                "xg": mom["xg_home"],
                "xcs": mom["xcs_home"],
                "xgc": mom["xgc_home"],
                "goals": fthg,
                "cs": int(ftag == 0),
                "gc_events": ftag // 2,
                "goals_against": ftag,
            }
        )
        rows.append(
            {
                **base,
                "side": "away",
                "team": str(r["AwayTeam"]),
                "opp": str(r["HomeTeam"]),
                "xg": mom["xg_away"],
                "xcs": mom["xcs_away"],
                "xgc": mom["xgc_away"],
                "goals": ftag,
                "cs": int(fthg == 0),
                "gc_events": fthg // 2,
                "goals_against": fthg,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    return out


def summarise(sides: pd.DataFrame) -> dict[str, Any]:
    xg = sides["xg"].to_numpy(float)
    goals = sides["goals"].to_numpy(float)
    xcs = sides["xcs"].to_numpy(float)
    cs = sides["cs"].to_numpy(float)
    xgc = sides["xgc"].to_numpy(float)
    gc = sides["gc_events"].to_numpy(float)

    return {
        "n_side_rows": int(len(sides)),
        "n_matches": int(sides[["date", "home", "away"]].drop_duplicates().shape[0]),
        "fit_ok_rate": float(sides["fit_ok"].mean()),
        "mean_sse": float(sides["sse"].mean()),
        "xg": {
            "pearson": _safe_corr(xg, goals),
            "spearman": _safe_corr(xg, goals, "spearman"),
            "mae": _mae(xg, goals),
            "rmse": _rmse(xg, goals),
            "mean_pred": float(xg.mean()),
            "mean_actual": float(goals.mean()),
            "bias": float(xg.mean() - goals.mean()),
        },
        "xcs": {
            "pearson": _safe_corr(xcs, cs),
            "brier": _brier(xcs, cs),
            "logloss": _logloss(xcs, cs),
            "mean_pred": float(xcs.mean()),
            "mean_actual": float(cs.mean()),
            "bias": float(xcs.mean() - cs.mean()),
            "calibration": _calibration_bins(xcs, cs, n_bins=8),
        },
        "xgc": {
            "pearson": _safe_corr(xgc, gc),
            "spearman": _safe_corr(xgc, gc, "spearman"),
            "mae": _mae(xgc, gc),
            "rmse": _rmse(xgc, gc),
            "mean_pred": float(xgc.mean()),
            "mean_actual": float(gc.mean()),
            "bias": float(xgc.mean() - gc.mean()),
        },
        # Naive baselines for context.
        "baselines": {
            "cs_brier_constant": _brier(np.full_like(cs, cs.mean()), cs),
            "xg_mae_constant": _mae(np.full_like(goals, goals.mean()), goals),
            "xgc_mae_constant": _mae(np.full_like(gc, gc.mean()), gc),
        },
    }


def write_report(season: str, summary: dict[str, Any], path: Path) -> None:
    xg, xcs, xgc = summary["xg"], summary["xcs"], summary["xgc"]
    b = summary["baselines"]
    lines = [
        f"# Scoreline moments vs actuals — {season}",
        "",
        "Football-data **opening** AH + OU → Dixon–Coles `M` (BTTS weight 0).",
        "Sides stacked (home and away). Actual CS = opponent scored 0; "
        "GC events = floor(GA/2).",
        "",
        f"- Matches fitted: **{summary['n_matches']}** ({summary['n_side_rows']} side-rows)",
        f"- Fit ok rate: **{summary['fit_ok_rate']:.3f}** · mean SSE(AH,OU): **{summary['mean_sse']:.2e}**",
        "",
        "## xG vs goals",
        f"- Pearson **{xg['pearson']:.3f}** · Spearman **{xg['spearman']:.3f}**",
        f"- MAE **{xg['mae']:.3f}** · RMSE **{xg['rmse']:.3f}** "
        f"(constant MAE {b['xg_mae_constant']:.3f})",
        f"- Mean pred/actual **{xg['mean_pred']:.3f} / {xg['mean_actual']:.3f}** "
        f"(bias {xg['bias']:+.3f})",
        "",
        "## xCS vs clean sheet",
        f"- Pearson **{xcs['pearson']:.3f}** · Brier **{xcs['brier']:.4f}** "
        f"(constant {b['cs_brier_constant']:.4f}) · logloss **{xcs['logloss']:.4f}**",
        f"- Mean pred/actual **{xcs['mean_pred']:.3f} / {xcs['mean_actual']:.3f}** "
        f"(bias {xcs['bias']:+.3f})",
        "",
        "## xGC vs GC events",
        f"- Pearson **{xgc['pearson']:.3f}** · Spearman **{xgc['spearman']:.3f}**",
        f"- MAE **{xgc['mae']:.3f}** · RMSE **{xgc['rmse']:.3f}** "
        f"(constant MAE {b['xgc_mae_constant']:.3f})",
        f"- Mean pred/actual **{xgc['mean_pred']:.3f} / {xgc['mean_actual']:.3f}** "
        f"(bias {xgc['bias']:+.3f})",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(season: str = "2025-26") -> dict[str, Any]:
    code = SEASON_FILES[season]
    fd_path = CACHE / code
    sides = fit_season_sides(fd_path)
    summary = summarise(sides)
    summary["season"] = season
    summary["source"] = str(fd_path.relative_to(ROOT))
    summary["markets"] = ["AH", "OU2.5"]
    summary["markets_missing"] = ["BTTS"]

    stem = f"scoreline_fd_moments_{season.replace('-', '_')}"
    csv_path = PROCESSED / f"{stem}.csv"
    json_path = PROCESSED / f"{stem}_summary.json"
    report_path = REPORTS / f"{stem}.md"

    sides.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(season, summary, report_path)

    return {
        "summary": summary,
        "csv": str(csv_path),
        "json": str(json_path),
        "report": str(report_path),
    }


def run_all() -> list[dict[str, Any]]:
    outs: list[dict[str, Any]] = []
    for season in SEASON_FILES:
        outs.append(run(season))
    return outs


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="FD AH+OU → M moments by season")
    p.add_argument(
        "--season",
        default="all",
        help="Season like 2025-26, or 'all' for every closed season",
    )
    args = p.parse_args()
    seasons = list(SEASON_FILES) if args.season == "all" else [args.season]
    for season in seasons:
        out = run(season)
        s = out["summary"]
        print(
            f"season {s['season']}  matches={s['n_matches']}  ok={s['fit_ok_rate']:.3f}  "
            f"xG r={s['xg']['pearson']:.3f}"
        )
        print(out["report"])


if __name__ == "__main__":
    main()
