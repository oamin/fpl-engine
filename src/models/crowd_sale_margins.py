"""Margins on the sales that made or gave back the second half.

Traces the four climbs that led their season. A sale is a played move
from a legal squad. The margin is the published transfer value minus the
hold, in the units the search already uses. The two calls are locked.
Neither one changes the hold margin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.crowd_opening_scores import (
    WEEK_CSV,
    load_openings,
    opening_state,
    prepare_season,
)
from src.models.ridge_multiseason import SEASONS
from src.models.season_climb_ft import HOLD_EPS, run_ft_season
from src.rules.fpl_2026 import half_for_gw

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SALE_CSV = PROCESSED / "crowd_sale_margins.csv"
REPORT_PATH = REPORTS / "crowd_sale_margins.md"

# The four climbs that led their season. Named before the margins are read.
LEADERS: tuple[tuple[str, str], ...] = (
    ("2022-23", "template"),
    ("2023-24", "template"),
    ("2024-25", "template"),
    ("2025-26", "next"),
)
FOCUS = ("2022-23", "template")
CONTROL = ("2023-24", "template")
UNDER = 2.5
OVER = 5.0
SHARE_GAP = 0.20
MIN_SALES = 8
GWS = list(range(1, 39))


def sale_rows(decisions: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Played moves from a legal squad. An illegal week is returned aside."""
    kept: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    for row in decisions:
        if row.get("role") != "move" or int(row.get("n_transfers", 0)) <= 0:
            continue
        record = {
            "gw": int(row["gw"]),
            "half": half_for_gw(int(row["gw"])),
            "margin": float(row["margin"]),
            "n_transfers": int(row["n_transfers"]),
            "hits": int(row["hits"]),
            "hold_legal": bool(row["hold_legal"]),
        }
        if record["hold_legal"]:
            kept.append(record)
        else:
            dropped.append(record)
    return pd.DataFrame(kept), pd.DataFrame(dropped)


def half_stats(margins: pd.Series) -> dict[str, float]:
    """Count, median, and the two locked shares. An empty half has no median."""
    values = pd.to_numeric(margins, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(values.size)
    if n == 0:
        return {"n": 0, "median": float("nan"), "share_under": float("nan"), "share_over": float("nan")}
    return {
        "n": n,
        "median": float(np.median(values)),
        "share_under": float(np.mean(values < UNDER)),
        "share_over": float(np.mean(values >= OVER)),
    }


def diagnose(focus: dict[str, float], control: dict[str, float]) -> str:
    """The two locked calls. Any other pattern is inconclusive.

    ``focus`` is the 2022/23 template second half. ``control`` is the
    2023/24 template second half. A half with fewer than eight sales
    cannot fire either call.
    """
    if int(focus["n"]) < MIN_SALES or int(control["n"]) < MIN_SALES:
        return "insufficient"
    share_gap = float(focus["share_under"]) - float(control["share_under"])
    loose = float(focus["median"]) < UNDER and share_gap + 1e-12 >= SHARE_GAP
    sure = float(focus["median"]) >= OVER and share_gap < SHARE_GAP
    if loose:
        return "bar loose"
    if sure:
        return "score sure and wrong"
    return "inconclusive"


def _blank_stats() -> dict[str, float]:
    return {"n": 0, "median": float("nan"), "share_under": float("nan"), "share_over": float("nan")}


def summarise(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per season, squad, half, and hit split. The call uses every sale."""
    rows: list[dict[str, object]] = []
    if sales.empty:
        return pd.DataFrame(rows)
    grouped = sales.groupby(["season", "squad", "half"], sort=False)
    for (season, squad, half), block in grouped:
        pieces = [("all", block), ("free", block.loc[block["hits"] == 0]), ("hit", block.loc[block["hits"] > 0])]
        for kind, part in pieces:
            stats = half_stats(part["margin"]) if len(part) else _blank_stats()
            rows.append(
                {
                    "season": season,
                    "squad": squad,
                    "half": half,
                    "kind": kind,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame) -> str:
    """Apply the call to the two locked second halves. Other rows stay context."""
    def cell(season: str, squad: str) -> dict[str, float]:
        block = summary.loc[
            (summary["season"] == season)
            & (summary["squad"] == squad)
            & (summary["half"] == "H2")
            & (summary["kind"] == "all")
        ]
        if block.empty:
            return _blank_stats()
        row = block.iloc[0]
        return {
            "n": float(row["n"]),
            "median": float(row["median"]),
            "share_under": float(row["share_under"]),
            "share_over": float(row["share_over"]),
        }

    return diagnose(cell(*FOCUS), cell(*CONTROL))


def _same_climb(weekly: pd.DataFrame, season: str, squad: str) -> str:
    """The trace has to be the climb already scored. A mismatch is a failure."""
    if not WEEK_CSV.exists():
        return "the scored climb is not on disk"
    stored = pd.read_csv(WEEK_CSV)
    prev = stored.loc[
        (stored["season"] == season) & (stored["squad"] == squad) & (stored["path"] == "climb")
    ]
    fresh = weekly.loc[weekly["method"] == "xp_ft"] if "method" in weekly.columns else weekly.iloc[0:0]
    if prev.empty or fresh.empty:
        return "a climb week is missing"
    same_weeks = sorted(int(g) for g in prev["gw"]) == sorted(int(g) for g in fresh["gw"])
    same_points = float(prev["xi_points_cap"].sum()) == float(fresh["xi_points_cap"].sum())
    same_transfers = int(prev["n_transfers"].sum()) == int(fresh["n_transfers"].sum())
    same_hits = int(prev["hits"].sum()) == int(fresh["hits"].sum())
    if same_weeks and same_points and same_transfers and same_hits:
        return ""
    return "the trace does not match the scored climb"


def trace_leader(season: str, squad: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """One leading climb, with the decision margin on every searched week."""
    code = dict(SEASONS)[season]
    feat, roster, _expected = prepare_season(season, code)
    openings = load_openings()
    block = openings.loc[
        (openings["season"] == season.replace("-", "_")) & (openings["squad"] == squad)
    ]
    opening = opening_state(season, block)
    decisions: list[dict] = []
    weekly = run_ft_season(
        feat,
        {"xp": "score_xp"},
        GWS,
        roster=roster,
        chips=None,
        opening=opening,
        freeze_horizon=False,
        decisions=decisions,
    )
    kept, dropped = sale_rows(decisions)
    for frame in (kept, dropped):
        if frame.empty:
            continue
        frame.insert(0, "squad", squad)
        frame.insert(0, "season", season)
    mismatch = _same_climb(weekly, season, squad)
    return kept, dropped, weekly, mismatch


def _pct(share: float) -> str:
    if share != share:
        return "—"
    return f"{100.0 * share:.0f}%"


def _num(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:.2f}"


def render_report(
    sales: pd.DataFrame,
    dropped: pd.DataFrame,
    summary: pd.DataFrame,
    call: str,
    *,
    review: str = "",
    mismatches: list[str] | None = None,
) -> str:
    """The two calls, then the four halves. Context rows do not move the call."""
    lines = [
        "# Sale margins on the leading climbs",
        "",
        "Gemini locked the trace before the margins were read "
        "([sale margins](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). "
        "The four climbs are the ones that led their season: the template in "
        "2022/23, 2023/24, and 2024/25, and Next in 2025/26. "
        "A sale is a played move from a legal squad. "
        f"The margin is the published transfer value minus the hold. "
        f"A legal squad needs {HOLD_EPS} to move. "
        "The value is the discounted three-week eleven, net of the hit and the switch penalty.",
        "",
        "The bar is loose only if the 2022/23 template second half has a median under "
        "2.5 and its share under 2.5 is at least 20 percentage points higher than the "
        "2023/24 template second half. The score was sure only if that median is at "
        "least 5 and the share under 2.5 is not 20 points higher. "
        "A half with fewer than 8 sales cannot fire either call. "
        "Anything else is inconclusive. 2024/25 and 2025/26 are reported beside the call. "
        "The hold margin stays 1.25.",
        "",
    ]
    if mismatches:
        lines.append("The trace does not match the scored climb: " + "; ".join(mismatches) + ".")
        lines.append("")
    if review:
        lines.extend([review.strip(), ""])
    else:
        lines.extend(["The diagnostic review of these margins is still open.", ""])
    lines.append(f"Call: **{call}**.")
    lines.append("")
    if not dropped.empty:
        bits = [
            f"{row.season} {row.squad} Gameweek {int(row.gw)}"
            for row in dropped.itertuples()
        ]
        lines.append("Illegal weeks dropped: " + ", ".join(bits) + ".")
        lines.append("")
    lines.append("| Season | Squad | Half | Sales | Median | Under 2.5 | At least 5 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    show = summary.loc[summary["kind"] == "all"] if not summary.empty else summary
    for _, row in show.iterrows():
        lines.append(
            f"| {row['season']} | {row['squad']} | {row['half']} | {int(row['n'])} | "
            f"{_num(float(row['median']))} | {_pct(float(row['share_under']))} | "
            f"{_pct(float(row['share_over']))} |"
        )
    lines.append("")
    lines.append("Free transfers and hits, same margins.")
    lines.append("")
    lines.append("| Season | Squad | Half | Kind | Sales | Median | Under 2.5 | At least 5 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    detail = summary.loc[summary["kind"] != "all"] if not summary.empty else summary
    for _, row in detail.iterrows():
        lines.append(
            f"| {row['season']} | {row['squad']} | {row['half']} | {row['kind']} | "
            f"{int(row['n'])} | {_num(float(row['median']))} | "
            f"{_pct(float(row['share_under']))} | {_pct(float(row['share_over']))} |"
        )
    lines.append("")
    lines.append(
        f"{len(sales)} legal sales. "
        "Week margins are in `data/processed/crowd_sale_margins.csv`."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    sales: pd.DataFrame,
    dropped: pd.DataFrame,
    *,
    review: str = "",
    mismatches: list[str] | None = None,
) -> str:
    summary = summarise(sales)
    call = "trace mismatch" if mismatches else verdict(summary)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    sales.to_csv(SALE_CSV, index=False)
    if not dropped.empty:
        dropped.to_csv(PROCESSED / "crowd_sale_margins_dropped.csv", index=False)
    REPORT_PATH.write_text(
        render_report(sales, dropped, summary, call, review=review, mismatches=mismatches),
        encoding="utf-8",
    )
    return call


def trace_all(leaders: tuple[tuple[str, str], ...] = LEADERS) -> str:
    frames: list[pd.DataFrame] = []
    dropped_frames: list[pd.DataFrame] = []
    mismatches: list[str] = []
    for season, squad in leaders:
        print(f"tracing {season} {squad}", flush=True)
        kept, dropped, _weekly, mismatch = trace_leader(season, squad)
        if mismatch:
            mismatches.append(f"{season} {squad}: {mismatch}")
        if not kept.empty:
            frames.append(kept)
        if not dropped.empty:
            dropped_frames.append(dropped)
    sales = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    dropped = pd.concat(dropped_frames, ignore_index=True) if dropped_frames else pd.DataFrame()
    return write_outputs(sales, dropped, mismatches=mismatches)


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    if len(sys.argv) == 3:
        season, squad = sys.argv[1], sys.argv[2]
        kept, dropped, _weekly, mismatch = trace_leader(season, squad)
        out = Path("/tmp") / "sale_margins"
        out.mkdir(parents=True, exist_ok=True)
        slug = f"{season}_{squad}"
        kept.to_csv(out / f"{slug}.csv", index=False)
        dropped.to_csv(out / f"{slug}_dropped.csv", index=False)
        (out / f"{slug}.mismatch").write_text(mismatch, encoding="utf-8")
        print(f"{season} {squad} sales {len(kept)} dropped {len(dropped)} {mismatch}", flush=True)
    else:
        print(trace_all())
