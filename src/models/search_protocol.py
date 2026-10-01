"""Two-tier search. Tier 1 is the fast XI climb on a full season.

Candidates live in ``experiments/matrix.json``. They are ranked by
cumulative XI points versus expected points. Spearman, MAE, and bias are
recorded and do not decide who advances.

The top ``advance_n`` architectures advance only when their best screen
point is not clearly behind (delta greater than ``-kill_gap``). The
baseline is the comparator and does not advance. A transfer-value
candidate is not ranked: the fast climb cannot see it.

Tier 2 runs the free-transfer climb on every grid point of those
architectures and records points, hits, and transfers. P* is the grid
point with the highest transfer-climb total. It is a winner only if it
then beats the paired expected-points climb by ``pass_margin`` on the
holdout season. The label is best of this search.

Writes:
  data/processed/search_tier1.csv
  data/processed/search_tier2.csv
  data/processed/search_holdout.csv
  reports/search_protocol.md
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.ridge_multiseason import SEASONS, build_one_season
from src.models.season_climb import summarize
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.models.sharpe_u import add_causal_sharpe_u
from src.models.stage_29_batch import fast_xi
from src.models.stage_30_followup import attach_ownership

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "experiments" / "matrix.json"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def load_matrix(path: Path | None = None) -> dict[str, Any]:
    raw = json.loads((path or MATRIX).read_text(encoding="utf-8"))
    if raw["advance_n"] < 1:
        raise ValueError("advance_n must be positive")
    if raw["kill_gap"] < 0 or raw["pass_margin"] < 0:
        raise ValueError("kill_gap and pass_margin must be non-negative")
    return raw


def apply_score(df: pd.DataFrame, formula: str, params: dict[str, Any]) -> pd.Series:
    """Decision-time score. Missing inputs stay missing."""
    xp = pd.to_numeric(df["score_xp"], errors="coerce")
    if formula == "column":
        return pd.to_numeric(df[str(params["column"])], errors="coerce")
    sigma = pd.to_numeric(df["sigma_xp"], errors="coerce")
    if formula == "linear_risk":
        return xp - float(params["lambda"]) * sigma
    if formula == "sharpe":
        return xp / (sigma + float(params["eps"]))
    if formula == "ownership":
        ow = pd.to_numeric(df["ow"], errors="coerce")
        return xp * (1.0 - float(params["weight"]) * ow)
    if formula == "blend":
        exp = pd.to_numeric(df["score_exp_points"], errors="coerce")
        alpha = float(params["alpha"])
        return alpha * xp + (1.0 - alpha) * exp
    if formula == "tail_risk":
        excess = (sigma - 3.0).clip(lower=0.0)
        return xp - float(params["lambda"]) * excess
    raise ValueError(f"unknown formula: {formula}")


def iter_grid(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    grid = candidate["params_grid"]
    if not grid:
        raise ValueError(f"{candidate['id']} has an empty params_grid")
    if isinstance(grid, list):
        return [dict(item) for item in grid]
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in product(*(grid[k] for k in keys))]


def _param_label(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    return ",".join(f"{key}={value}" for key, value in params.items())


def score_diagnostics(
    feat: pd.DataFrame,
    gws: list[int],
    column: str,
) -> dict[str, float]:
    """Pooled player-week fit. Diagnostic only; it does not rank candidates."""
    sl = feat.loc[feat["gw"].isin(gws) & feat["eligible"].astype(bool), [column, "total_points"]]
    sl = sl.replace([np.inf, -np.inf], np.nan).dropna()
    pred = sl[column].astype(float)
    actual = sl["total_points"].astype(float)
    if len(sl) < 3 or pred.nunique(dropna=True) < 2 or actual.nunique(dropna=True) < 2:
        return {"spearman": float("nan"), "mae": float("nan"), "bias": float("nan"), "n": float(len(sl))}
    return {
        "spearman": float(pred.corr(actual, method="spearman")),
        "mae": float((pred - actual).abs().mean()),
        "bias": float((pred - actual).mean()),
        "n": float(len(sl)),
    }


def select_architectures(
    tier1: pd.DataFrame,
    *,
    advance_n: int,
    kill_gap: float,
    baseline_id: str = "xp",
) -> pd.DataFrame:
    """Top architectures by their best fast-XI delta, if not clearly behind."""
    scored = tier1.loc[tier1["family"] != "transfer_value"].copy()
    if scored.empty:
        return scored
    best = (
        scored.sort_values(["delta_vs_xp", "candidate"], ascending=[False, True])
        .groupby("candidate", as_index=False)
        .head(1)
    )
    alive = best.loc[
        (best["candidate"] != baseline_id) & (best["delta_vs_xp"] > -float(kill_gap))
    ]
    return alive.sort_values(["delta_vs_xp", "candidate"], ascending=[False, True]).head(advance_n)


def _fd_code(season: str) -> str:
    for name, code in SEASONS:
        if name == season:
            return code
    raise KeyError(season)


def _gws(feat: pd.DataFrame, spec: dict[str, Any]) -> list[int]:
    return [
        g
        for g in sorted(int(x) for x in feat["gw"].unique())
        if spec["gw_start"] <= g <= spec["gw_end"]
    ]


def prepare_season(season: str) -> pd.DataFrame:
    feat = build_one_season(season, _fd_code(season))
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    feat = add_causal_sharpe_u(feat)
    return attach_ownership(feat, season)


def _attach_candidates(
    feat: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    only: set[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = feat.copy()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["family"] == "transfer_value":
            continue
        if only is not None and candidate["id"] not in only and candidate["id"] != "xp":
            continue
        for index, params in enumerate(iter_grid(candidate)):
            column = f"search__{candidate['id']}__{index}"
            out[column] = apply_score(out, candidate["formula"], params)
            rows.append(
                {
                    "candidate": candidate["id"],
                    "family": candidate["family"],
                    "formula": candidate["formula"],
                    "params": _param_label(params),
                    "column": column,
                }
            )
    return out, rows


def run_tier1(feat: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gws = _gws(feat, spec)
    scored, meta = _attach_candidates(feat, spec["candidates"])
    cols = {row["column"]: row["column"] for row in meta}
    weekly = fast_xi(scored, gws, cols)
    summary = summarize(weekly)
    totals = summary.set_index("method")["total_points"]
    xp_cols = [row["column"] for row in meta if row["candidate"] == "xp"]
    if len(xp_cols) != 1:
        raise RuntimeError("matrix needs exactly one xp baseline")
    base = float(totals[xp_cols[0]])
    rows = []
    for row in meta:
        diag = score_diagnostics(scored, gws, row["column"])
        total = float(totals[row["column"]])
        rows.append(
            {
                **row,
                "xi_points": total,
                "delta_vs_xp": total - base,
                "spearman": diag["spearman"],
                "mae": diag["mae"],
                "bias": diag["bias"],
                "n": diag["n"],
                "mean_transfers": float("nan"),
                "hits": float("nan"),
            }
        )
    return pd.DataFrame(rows), scored


def _ft_stats(weekly: pd.DataFrame, method: str) -> dict[str, float]:
    frame = weekly.loc[weekly["method"] == method]
    if frame.empty:
        return {"xi_points": float("nan"), "mean_transfers": float("nan"), "hits": float("nan")}
    return {
        "xi_points": float(frame["xi_points_cap"].sum()),
        "mean_transfers": float(frame["n_transfers"].mean()),
        "hits": float(frame["hits"].sum()),
    }


def run_tier2(
    feat: pd.DataFrame,
    spec: dict[str, Any],
    tier1: pd.DataFrame,
    advanced: pd.DataFrame,
) -> pd.DataFrame:
    """Free-transfer climb for every grid point of the advanced architectures."""
    if advanced.empty:
        return pd.DataFrame()
    names = set(advanced["candidate"])
    gws = _gws(feat, spec)
    roster = load_vaastav_roster(spec["screen_season"])
    print("  tier2 xp_ft…", flush=True)
    xp_weekly = run_ft_season(
        feat, {"xp": "score_xp"}, gws, roster=roster, horizon=HORIZON
    )
    xp_total = _ft_stats(xp_weekly, "xp_ft")["xi_points"]
    frames = [xp_weekly]
    rows = []
    grid = tier1.loc[tier1["candidate"].isin(names)]
    for record in grid.itertuples(index=False):
        label = f"{record.candidate}__{record.params}".replace("=", "").replace(",", "_")
        print(f"  tier2 {label}…", flush=True)
        weekly = run_ft_season(
            feat,
            {label: record.column},
            gws,
            roster=roster,
            horizon=HORIZON,
        )
        frames.append(weekly)
        stats = _ft_stats(weekly, f"{label}_ft")
        rows.append(
            {
                "candidate": record.candidate,
                "params": record.params,
                "column": record.column,
                "xi_points": stats["xi_points"],
                "delta_vs_xp": stats["xi_points"] - xp_total,
                "mean_transfers": stats["mean_transfers"],
                "hits": stats["hits"],
                "spearman": record.spearman,
                "mae": record.mae,
                "bias": record.bias,
            }
        )
    return pd.DataFrame(rows)


def choose_p_star(tier2: pd.DataFrame) -> pd.Series | None:
    if tier2.empty:
        return None
    ordered = tier2.sort_values(
        ["xi_points", "mean_transfers", "candidate"],
        ascending=[False, True, True],
    )
    return ordered.iloc[0]


def run_holdout(spec: dict[str, Any], winner: pd.Series, screen_candidates: list[dict]) -> pd.DataFrame:
    season = spec["holdout_season"]
    print(f"Holdout {season}…", flush=True)
    feat = prepare_season(season)
    parent = next(item for item in screen_candidates if item["id"] == winner["candidate"])
    params = next(
        p
        for p in iter_grid(parent)
        if _param_label(p) == winner["params"]
    )
    feat["score_holdout"] = apply_score(feat, parent["formula"], params)
    gws = _gws(feat, spec)
    roster = load_vaastav_roster(season)
    xp = run_ft_season(feat, {"xp": "score_xp"}, gws, roster=roster, horizon=HORIZON)
    arm = run_ft_season(
        feat, {"winner": "score_holdout"}, gws, roster=roster, horizon=HORIZON
    )
    weekly = pd.concat([xp, arm], ignore_index=True)
    weekly.insert(0, "season", season)
    return weekly


def write_report(
    path: Path,
    spec: dict[str, Any],
    tier1: pd.DataFrame,
    advanced: pd.DataFrame,
    tier2: pd.DataFrame,
    holdout: pd.DataFrame | None,
) -> None:
    lines = [
        "# Search protocol",
        "",
        f"Screen {spec['screen_season']}, gameweeks {spec['gw_start']}–{spec['gw_end']}. "
        f"Ranked by fast XI points versus xp. A candidate is clearly behind at "
        f"{spec['kill_gap']:.0f} points. Top {spec['advance_n']} architectures go to "
        "the free-transfer grid. Spearman, MAE, and bias do not decide advancement.",
        "",
        f"Holdout season: {spec['holdout_season']}. "
        f"A winner must beat that season's xp climb by {spec['pass_margin']:.0f}.",
        "",
        "## Tier 1",
        "",
        "| candidate | params | XI points | vs xp | spearman | MAE | bias |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    ordered = tier1.sort_values("delta_vs_xp", ascending=False)
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.params} | {row.xi_points:.0f} | "
            f"{row.delta_vs_xp:+.0f} | {row.spearman:.3f} | {row.mae:.2f} | {row.bias:.2f} |"
        )
    if advanced.empty:
        lines += ["", "No architecture advanced. Tier 2 was not run.", ""]
    else:
        names = ", ".join(advanced["candidate"].tolist())
        lines += ["", f"Advanced: **{names}**.", ""]
    lines += [
        "## Tier 2",
        "",
        "| candidate | params | FT points | vs xp_ft | transfers/GW | hits |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if tier2.empty:
        lines.append("| — | — | — | — | — | — |")
    else:
        for row in tier2.sort_values("xi_points", ascending=False).itertuples(index=False):
            lines.append(
                f"| {row.candidate} | {row.params} | {row.xi_points:.0f} | "
                f"{row.delta_vs_xp:+.0f} | {row.mean_transfers:.2f} | {row.hits:.0f} |"
            )
    lines += ["", "## Holdout", ""]
    if holdout is None or holdout.empty:
        lines.append("Not run. Nothing on the screen-season transfer climb was ahead of xp.")
    else:
        summary = summarize(holdout)
        xp = float(summary.loc[summary["method"] == "xp_ft", "total_points"].iloc[0])
        arm = float(summary.loc[summary["method"] == "winner_ft", "total_points"].iloc[0])
        delta = arm - xp
        season = holdout["season"].iloc[0]
        if delta >= spec["pass_margin"]:
            verdict = (
                f"WINNER on this protocol: {delta:+.0f} on {season}. "
                "Best of this search, and it cleared the holdout bar."
            )
        else:
            verdict = (
                f"NO WINNER. Holdout delta on {season} is {delta:+.0f} "
                f"(bar +{spec['pass_margin']:.0f})."
            )
        lines += [
            f"| method | total |",
            f"|---|---:|",
            f"| xp_ft | {xp:.0f} |",
            f"| P* | {arm:.0f} |",
            "",
            f"**{verdict}**",
        ]
    lines += [
        "",
        "Transfer-value candidates are listed in the matrix and are not ranked here.",
        "",
        "- `data/processed/search_tier1.csv`",
        "- `data/processed/search_tier2.csv`",
        "- `data/processed/search_holdout.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(matrix_path: Path | None = None) -> dict[str, Any]:
    spec = load_matrix(matrix_path)
    print(f"Building {spec['screen_season']}…", flush=True)
    feat = prepare_season(spec["screen_season"])
    print("Tier 1…", flush=True)
    tier1, scored = run_tier1(feat, spec)
    advanced = select_architectures(
        tier1, advance_n=int(spec["advance_n"]), kill_gap=float(spec["kill_gap"])
    )
    print(advanced.to_string(index=False) if len(advanced) else "no advancers", flush=True)
    print("Tier 2…", flush=True)
    tier2 = run_tier2(scored, spec, tier1, advanced)
    holdout = None
    winner = choose_p_star(tier2)
    if winner is not None and float(winner["delta_vs_xp"]) > 0:
        holdout = run_holdout(spec, winner, spec["candidates"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tier1.to_csv(PROCESSED / "search_tier1.csv", index=False)
    tier2.to_csv(PROCESSED / "search_tier2.csv", index=False)
    if holdout is not None:
        holdout.to_csv(PROCESSED / "search_holdout.csv", index=False)
    write_report(REPORTS / "search_protocol.md", spec, tier1, advanced, tier2, holdout)
    print(f"Wrote {REPORTS / 'search_protocol.md'}", flush=True)
    return {"tier1": tier1, "advanced": advanced, "tier2": tier2, "holdout": holdout}


if __name__ == "__main__":
    run()
