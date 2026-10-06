"""The two readings locked in reports/decision_margin_plan.md.

Reading A bins the chip signings the rebuild left out. The ranking call
uses only the swaps that were not blocked by club or price. Reading B
is the historical rank-band bias. Neither changes the score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.reset_gap import PROCESSED, REPORTS

OUTSIDE_POINTS = -229.0
HELD_POINTS = -100.0
NARROW = 0.5
HOLD = 1.25
SHARE = 0.5
CURSE_GAP = 0.25
FIRST_GW = 6
LAST_GW = 38
BANDS = ("1-5", "6-10", "11-20", "rest")
PLAYERS = PROCESSED / "chip_lead_players_gw15.csv"
PLAN = REPORTS / "decision_margin_plan.md"


def margin(gap: float) -> float:
    """Model player minus human. The stored gap is human minus model."""
    return -float(gap)


def bin_name(value: float) -> str:
    """Closed on the right at 0.5 and at 1.25. Above 1.25 is its own bin."""
    if value <= 0:
        return "<=0"
    if value <= NARROW:
        return "0-0.5"
    if value <= HOLD:
        return "0.5-1.25"
    return ">1.25"


def ranking_call(shares: dict[str, float]) -> str:
    """Half the unconstrained weight, or inconclusive. The bar is not lowered."""
    if shares.get("0-0.5", 0.0) >= SHARE:
        return "small-margin"
    if shares.get(">1.25", 0.0) >= SHARE:
        return "wide-margin"
    return "inconclusive"


def curse_call(seasons: list[dict[str, float]]) -> str:
    """Positive top-band bias, and 0.25 above the eligible bias, in every season."""
    if len(seasons) != 4:
        return "parked"
    for season in seasons:
        top = float(season["top_bias"])
        base = float(season["eligible_bias"])
        if top <= 0 or top < base + CURSE_GAP:
            return "parked"
    return "curse"


def _share(points: float, total: float) -> float:
    if abs(total) < 1e-9:
        raise RuntimeError("a bin share needs a non-zero weight")
    return float(points / total)


def read_signings(frame: pd.DataFrame) -> dict[str, Any]:
    """The 64 left out, split by the stored block. Reference rows stay out."""
    rows = frame.loc[
        (frame["kind"] == "chip_squad")
        & (frame["tag"] == "eligible")
        & (frame["group"] != "reference")
    ].copy()
    outside = rows.loc[rows["place"] == "out"]
    held = rows.loc[rows["place"] == "in"]
    if len(outside) != 64 or abs(float(outside["points"].sum()) - OUTSIDE_POINTS) > 1e-6:
        raise RuntimeError("the outside signings are not the published 64")
    if len(held) != 19 or abs(float(held["points"].sum()) - HELD_POINTS) > 1e-6:
        raise RuntimeError("the held signings are not the published 19")
    constrained = outside.loc[outside["block"].isin(["club", "price"])]
    free = outside.loc[outside["block"] == "neither"]
    other = outside.loc[~outside["block"].isin(["club", "price", "neither"])]
    if len(other):
        raise RuntimeError("a signing has a block other than club, price, or neither")
    total = float(free["points"].sum())
    bins: dict[str, dict[str, float]] = {
        name: {"n": 0, "points": 0.0} for name in ("<=0", "0-0.5", "0.5-1.25", ">1.25")
    }
    for row in free.itertuples(index=False):
        name = bin_name(margin(float(row.gap)))
        bins[name]["n"] += 1
        bins[name]["points"] += float(row.points)
    shares = {name: _share(float(block["points"]), total) for name, block in bins.items()}
    constrained_points = float(constrained["points"].sum())
    outside_points = float(outside["points"].sum())
    return {
        "n_out": int(len(outside)),
        "points_out": outside_points,
        "n_held": int(len(held)),
        "points_held": float(held["points"].sum()),
        "n_constrained": int(len(constrained)),
        "points_constrained": constrained_points,
        "constrained_share": _share(constrained_points, outside_points),
        "n_free": int(len(free)),
        "points_free": total,
        "bins": bins,
        "shares": shares,
        "top": "constraint",
        "ranking": ranking_call(shares),
    }


def _band(rank: int) -> str:
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    return "rest"


def rank_rows(feat: pd.DataFrame, history: dict[str, dict[int, float]], season: str) -> list[dict[str, Any]]:
    """Eligible rows from week 6, ranked within position. An equal score follows the player id."""
    from src.models.close_pair_shape import _candidates

    candidates = _candidates(feat)
    candidates = candidates.loc[candidates["gw"].between(FIRST_GW, LAST_GW)]
    rows: list[dict[str, Any]] = []
    for (_gw, _position), block in candidates.groupby(["gw", "position"], sort=False):
        kept = []
        for row in block.itertuples(index=False):
            played = history.get(str(row.element), {}).get(int(row.gw))
            if played is None:
                continue
            kept.append(row)
        kept.sort(key=lambda row: (-float(row.score_xp), str(row.player_id)))
        for rank, row in enumerate(kept, start=1):
            rows.append(
                {
                    "season": season,
                    "gw": int(row.gw),
                    "position": str(row.position),
                    "player_id": str(row.player_id),
                    "rank": rank,
                    "band": _band(rank),
                    "score_xp": float(row.score_xp),
                    "points": float(history[str(row.element)][int(row.gw)]),
                }
            )
    return rows


def _bias(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise RuntimeError("a bias needs at least one row")
    predicted = sum(float(row["score_xp"]) for row in rows) / len(rows)
    actual = sum(float(row["points"]) for row in rows) / len(rows)
    return float(predicted - actual)


def season_bands(rows: list[dict[str, Any]], season: str) -> dict[str, Any]:
    block = [row for row in rows if row["season"] == season]
    bands = {}
    for name in BANDS:
        part = [row for row in block if row["band"] == name]
        bands[name] = {
            "n": len(part),
            "bias": None if not part else _bias(part),
            "mean_xp": None if not part else sum(float(row["score_xp"]) for row in part) / len(part),
            "mean_points": None if not part else sum(float(row["points"]) for row in part) / len(part),
        }
    return {
        "season": season,
        "n": len(block),
        "eligible_bias": _bias(block),
        "top_bias": bands["1-5"]["bias"],
        "bands": bands,
    }


def summarise_bands(rows: list[dict[str, Any]], seasons: list[str]) -> dict[str, Any]:
    built = [season_bands(rows, season) for season in seasons]
    compact = [
        {"top_bias": float(row["top_bias"]), "eligible_bias": float(row["eligible_bias"])}
        for row in built
    ]
    return {"seasons": built, "call": curse_call(compact)}


def _fmt_points(value: float) -> str:
    return f"{value:.0f}"


def _fmt_share(value: float) -> str:
    if abs(value) < 5e-3:
        return "0.00"
    return f"{value:.2f}"


def write_report(path: Path, signing: dict[str, Any], bands: dict[str, Any]) -> None:
    lines = [
        "# Decision margin",
        "",
        "The two readings locked in `reports/decision_margin_plan.md`, counted after the lock. The score is not changed. The pair's realised points are not on the stored signing row, so they do not enter this count and they do not choose the call.",
        "",
        "## Reading A",
        "",
        "The 64 signings the rebuild left out. The pair is that rebuild's lowest score at the position. The margin is that score minus the human's. Club or price is the constraint. The ranking bins are only the rows tagged neither.",
        "",
        (
            f"Outside: {signing['n_out']} signings, {_fmt_points(signing['points_out'])}. "
            f"Held by the rebuild: {signing['n_held']} signings, {_fmt_points(signing['points_held'])}. "
            f"Constrained: {signing['n_constrained']} signings, {_fmt_points(signing['points_constrained'])}, "
            f"share {signing['constrained_share']:.2f}. "
            f"Unconstrained: {signing['n_free']} signings, {_fmt_points(signing['points_free'])}."
        ),
        "",
        "The top-level call on the outside points is constraint. The share is already past a half.",
        "",
        "| Margin | Signings | Points | Share of the unconstrained points |",
        "|---|---:|---:|---:|",
    ]
    for name in ("<=0", "0-0.5", "0.5-1.25", ">1.25"):
        block = signing["bins"][name]
        lines.append(
            f"| {name} | {int(block['n'])} | {_fmt_points(block['points'])} | {_fmt_share(signing['shares'][name])} |"
        )
    lines.extend(
        [
            "",
            f"The ranking call is {signing['ranking']}.",
            "",
            "## Reading B",
            "",
            "Seasons 2022/23 through 2025/26. Weeks 6 to 38. An eligible row has at least three prior appearances and expected minutes of at least 45. Rank is within position that week. An equal score keeps the earlier player id as the higher rank. Bias is mean score minus mean points. The top band is ranks 1 to 5.",
            "",
            f"The winner's-curse claim is {bands['call']}.",
            "",
            "| Season | Eligible rows | Eligible bias | Ranks 1–5 bias | Gap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for season in bands["seasons"]:
        gap = float(season["top_bias"]) - float(season["eligible_bias"])
        lines.append(
            f"| {season['season'].replace('-', '/')} | {season['n']} | {season['eligible_bias']:.2f} | "
            f"{season['top_bias']:.2f} | {gap:.2f} |"
        )
    lines.extend(["", "Bands:", ""])
    lines.append("| Season | Band | Rows | Mean score | Mean points | Bias |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for season in bands["seasons"]:
        for name in BANDS:
            band = season["bands"][name]
            lines.append(
                f"| {season['season'].replace('-', '/')} | {name} | {band['n']} | "
                f"{band['mean_xp']:.2f} | {band['mean_points']:.2f} | {band['bias']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            _reading_sentence(signing, bands),
            "",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _reading_sentence(signing: dict[str, Any], bands: dict[str, Any]) -> str:
    ranking = signing["ranking"]
    curse = bands["call"]
    if ranking == "small-margin" and curse == "parked":
        return (
            "The accessible disagreements are small-margin, and the top band does not clear the curse bar. "
            "The score stays as it is. A residual on the decision boundary is not opened."
        )
    if ranking == "wide-margin":
        return (
            "The accessible disagreements are wide-margin. The score stays as it is. "
            "No feature is added from these five weeks."
        )
    if ranking == "inconclusive":
        return (
            "The accessible disagreements do not gather in one bin. The score stays as it is. "
            f"The winner's-curse claim is {curse}."
        )
    return (
        f"The ranking call is {ranking}. The winner's-curse claim is {curse}. The score stays as it is."
    )


def _band_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    """Count the locked readings. Does not retune."""
    from src.models.close_pair_shape import _sheet_points
    from src.models.ridge_multiseason import SEASONS, build_one_season

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    signing = read_signings(pd.read_csv(PLAYERS))
    names = [season for season, _code in SEASONS]
    rows: list[dict[str, Any]] = []
    for season, code in SEASONS:
        print(f"margin {season}", flush=True)
        feat = build_one_season(season, code)
        found = rank_rows(feat, _sheet_points(season), season)
        print(f"  rows {len(found)}", flush=True)
        rows.extend(found)
    bands = summarise_bands(rows, names)
    _band_frame(rows).to_csv(PROCESSED / "decision_margin_bands.csv", index=False)
    write_report(REPORTS / "decision_margin.md", signing, bands)
    return {
        "ranking": signing["ranking"],
        "constrained_share": signing["constrained_share"],
        "points_free": signing["points_free"],
        "curse": bands["call"],
    }


if __name__ == "__main__":
    print(run())
