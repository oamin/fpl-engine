"""One closed gameweek: does score_xp line up with the scraped xP column?

The published −0.35 figure is a gap between two rank correlations with
points. It is not the correlation of the two forecasts. This check is a
diagnostic. It is not a benchmark and it does not enter the squad.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.rules.fpl_2026 import SQUAD_QUOTA


def _corr(left: np.ndarray, right: np.ndarray, kind: str) -> float | None:
    if left.size < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    stat = spearmanr if kind == "spearman" else pearsonr
    value = float(stat(left, right).statistic)
    if not np.isfinite(value):
        return None
    return value


def alignment_report(frame: pd.DataFrame, gw: int) -> dict[str, Any]:
    """Key alignment and the correlation of the two forecasts on one gameweek."""
    block = frame.loc[pd.to_numeric(frame["gw"], errors="coerce") == int(gw)].copy()
    if block.empty:
        raise RuntimeError(f"gameweek {gw} has no rows")
    if "fixture_id" not in block.columns:
        block["fixture_id"] = np.arange(len(block)).astype(str)
    keys = block.groupby(["player_id", "fixture_id"], sort=False).size()
    duplicate_keys = int((keys > 1).sum())
    per_player = block.groupby("player_id", sort=False).size()
    for column in ("score_xp", "official_xp", "total_points"):
        block[column] = pd.to_numeric(block[column], errors="coerce")
    ordered = block.sort_values(["player_id", "fixture_id"], kind="mergesort")
    base = ordered.groupby("player_id", as_index=False).first()
    summed = ordered.groupby("player_id", as_index=False)[
        ["score_xp", "official_xp", "total_points"]
    ].sum()
    keep = [column for column in base.columns if column not in summed.columns or column == "player_id"]
    collapsed = base[keep].merge(summed, on="player_id", how="left")
    both = collapsed["score_xp"].notna() & collapsed["official_xp"].notna()
    paired = collapsed.loc[both]
    score = paired["score_xp"].to_numpy(float)
    official = paired["official_xp"].to_numpy(float)
    points = paired["total_points"].to_numpy(float)
    spearman = _corr(score, official, "spearman")
    pearson = _corr(score, official, "pearson")
    negated = _corr(-score, official, "spearman")
    score_points = _corr(score, points, "spearman")
    official_points = _corr(official, points, "spearman")
    positions_ok = bool(paired["position"].isin(list(SQUAD_QUOTA)).all()) if len(paired) else False
    gw_ok = bool((paired["gw"] == int(gw)).all()) if len(paired) else False
    rank_gap = None
    if score_points is not None and official_points is not None:
        rank_gap = float(score_points - official_points)
    sign_flip = False
    if spearman is not None and negated is not None:
        sign_flip = negated > spearman
    return {
        "gw": int(gw),
        "n_rows": int(len(block)),
        "n_players": int(per_player.size),
        "max_rows_per_player": int(per_player.max()) if len(per_player) else 0,
        "duplicate_player_fixture": duplicate_keys,
        "n_paired": int(len(paired)),
        "same_row": bool(duplicate_keys == 0 and len(collapsed) == int(per_player.size)),
        "spearman_score_vs_official": spearman,
        "pearson_score_vs_official": pearson,
        "spearman_negated_score_vs_official": negated,
        "sign_flip_fits_better": sign_flip,
        "spearman_score_vs_points": score_points,
        "spearman_official_vs_points": official_points,
        "rank_gap_score_minus_official": rank_gap,
        "positions_ok": positions_ok,
        "gw_ok": gw_ok,
        "_paired": paired,
    }


def eligible_rank_gap_by_week(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Weekly piece of the withdrawn −0.35.

    Eligible players only. Within each position, Spearman(score_xp, points)
    minus Spearman(scraped xP, points), then the mean of those positions.
    An unfilled scrape is undefined. This is not a benchmark.
    """
    from src.eval.slices import POSITIONS, _spearman

    work = frame.loc[frame["eligible"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    if work.empty or "official_xp" not in work.columns:
        return rows
    season = str(work["season"].iloc[0]) if "season" in work.columns else ""
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    for gw, block in work.dropna(subset=["gw"]).groupby("gw", sort=True):
        order = [column for column in ("date", "fixture_id", "player_id") if column in block.columns]
        ordered = block.sort_values(order or ["player_id"], kind="mergesort")
        base = ordered.groupby("player_id", as_index=False).first()
        summed = ordered.groupby("player_id", as_index=False)[
            ["total_points", "score_xp", "official_xp"]
        ].sum()
        keep = [column for column in base.columns if column not in summed.columns or column == "player_id"]
        players = base[keep].merge(summed, on="player_id", how="left")
        official = pd.to_numeric(players["official_xp"], errors="coerce")
        filled = bool(official.notna().any() and float(official.max()) > 0.0)
        score_levels: list[float] = []
        scraped_levels: list[float] = []
        if filled:
            for position in POSITIONS:
                group = players.loc[players["position"] == position]
                y = pd.to_numeric(group["total_points"], errors="coerce").to_numpy(float)
                score = pd.to_numeric(group["score_xp"], errors="coerce").to_numpy(float)
                scraped = pd.to_numeric(group["official_xp"], errors="coerce").to_numpy(float)
                mask = np.isfinite(y) & np.isfinite(score) & np.isfinite(scraped)
                rho_score = _spearman(y[mask], score[mask])
                rho_scraped = _spearman(y[mask], scraped[mask])
                if np.isfinite(rho_score) and np.isfinite(rho_scraped):
                    score_levels.append(float(rho_score))
                    scraped_levels.append(float(rho_scraped))
        gap = float(np.mean(np.subtract(score_levels, scraped_levels))) if score_levels else None
        rows.append(
            {
                "season": season,
                "gw": int(gw),
                "rank_gap": gap,
                "rho_score": float(np.mean(score_levels)) if score_levels else None,
                "rho_scraped": float(np.mean(scraped_levels)) if scraped_levels else None,
                "negative": bool(gap is not None and gap < 0.0),
                "undefined": gap is None,
                "n_positions": int(len(score_levels)),
            }
        )
    return rows


def spearman_by_week(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Spearman(score_xp, scraped xP) on every gameweek. A diagnostic, not a benchmark."""
    rows: list[dict[str, Any]] = []
    gws = sorted(int(gw) for gw in pd.to_numeric(frame["gw"], errors="coerce").dropna().unique())
    for gw in gws:
        report = alignment_report(frame, gw)
        report.pop("_paired", None)
        spearman = report["spearman_score_vs_official"]
        rows.append(
            {
                "season": str(frame["season"].iloc[0]) if "season" in frame.columns else "",
                "gw": gw,
                "n_players": report["n_players"],
                "n_paired": report["n_paired"],
                "max_rows_per_player": report["max_rows_per_player"],
                "duplicate_player_fixture": report["duplicate_player_fixture"],
                "spearman": spearman,
                "negative": bool(spearman is not None and spearman < 0.0),
                "undefined": spearman is None,
            }
        )
    return rows


def save_scatter(frame: pd.DataFrame, gw: int, path: Path) -> dict[str, Any]:
    """Write the scatter and return the report without the paired frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = alignment_report(frame, gw)
    paired = report.pop("_paired")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(
        paired["official_xp"].to_numpy(float),
        paired["score_xp"].to_numpy(float),
        s=12,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_xlabel("scraped xP")
    ax.set_ylabel("score_xp")
    ax.set_title(f"GW{int(gw)} score_xp against scraped xP")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return report
