"""Decision-relevant slices. The bootstrap still resamples gameweeks.

Eligible rows only. Rank is Spearman within position. The top 15 of two
scores are paired on the players both lists name.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.eval.gates import COMPARISONS, comparison_key, gaussian_log_score
from src.eval.official_xp import usable_gameweeks

POSITIONS = ("GKP", "DEF", "MID", "FWD")


def _spearman(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) < 3 or np.unique(y).size < 2 or np.unique(score).size < 2:
        return float("nan")
    rho = spearmanr(y, score).correlation
    return float(rho) if rho is not None else float("nan")


def slice_rows(frame: pd.DataFrame, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """One gameweek delta per comparison, for rank and for the top-15 intersection."""
    spec = protocol["decision_slice"]
    top_n = int(spec["top_n"])
    min_both = int(spec["min_intersection"])
    sigma = float(protocol["sigma"])
    gw = pd.to_numeric(frame["gw"], errors="coerce")
    window = frame.loc[
        (gw >= int(protocol["gw_start"])) & (gw <= int(protocol["gw_end"])) & frame["eligible"]
    ].copy()
    if window.empty:
        return []
    filled = usable_gameweeks(window)
    season = str(window["season"].iloc[0])
    rows: list[dict[str, Any]] = []
    for left, right in COMPARISONS:
        needs_official = "score_official_xp" in (left, right)
        for gameweek, block in window.groupby("gw", sort=True):
            if needs_official and int(gameweek) not in filled:
                continue
            ordered = block.sort_values(["date", "fixture_id"], kind="mergesort")
            base = ordered.groupby("player_id", as_index=False).first()
            sums = ordered.groupby("player_id", as_index=False)[
                ["total_points", "score_xp", "score_exp_points", "score_official_xp"]
            ].sum()
            keep = [col for col in base.columns if col not in sums.columns or col == "player_id"]
            players = base[keep].merge(sums, on="player_id", how="left")
            rank_parts: list[float] = []
            top_parts: list[float] = []
            for position in POSITIONS:
                group = players.loc[players["position"] == position].copy()
                y = pd.to_numeric(group["total_points"], errors="coerce")
                a = pd.to_numeric(group[left], errors="coerce")
                b = pd.to_numeric(group[right], errors="coerce")
                ok = y.notna() & a.notna() & b.notna()
                group = group.loc[ok].copy()
                if len(group) < 3:
                    continue
                group["_y"] = y.loc[ok].to_numpy(float)
                group["_a"] = a.loc[ok].to_numpy(float)
                group["_b"] = b.loc[ok].to_numpy(float)
                rho_a = _spearman(group["_y"].to_numpy(float), group["_a"].to_numpy(float))
                rho_b = _spearman(group["_y"].to_numpy(float), group["_b"].to_numpy(float))
                if np.isfinite(rho_a) and np.isfinite(rho_b):
                    rank_parts.append(rho_a - rho_b)
                if len(group) < top_n:
                    continue
                ids_a = set(group.nlargest(top_n, "_a")["player_id"].astype(str))
                ids_b = set(group.nlargest(top_n, "_b")["player_id"].astype(str))
                both = ids_a & ids_b
                if len(both) < min_both:
                    continue
                chosen = group.loc[group["player_id"].astype(str).isin(both)]
                log_a = gaussian_log_score(
                    chosen["total_points"].to_numpy(float),
                    pd.to_numeric(chosen[left], errors="coerce").to_numpy(float),
                    sigma,
                )
                log_b = gaussian_log_score(
                    chosen["total_points"].to_numpy(float),
                    pd.to_numeric(chosen[right], errors="coerce").to_numpy(float),
                    sigma,
                )
                top_parts.append(float((log_a - log_b).mean()))
            if rank_parts:
                rows.append(
                    {
                        "season": season,
                        "gw": int(gameweek),
                        "comparison": comparison_key(left, right),
                        "slice": "spearman",
                        "delta": float(np.mean(rank_parts)),
                        "n_positions": int(len(rank_parts)),
                    }
                )
            if top_parts:
                rows.append(
                    {
                        "season": season,
                        "gw": int(gameweek),
                        "comparison": comparison_key(left, right),
                        "slice": "top15",
                        "delta": float(np.mean(top_parts)),
                        "n_positions": int(len(top_parts)),
                    }
                )
    return rows
