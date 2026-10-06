"""Buy gate for a player with one or two appearances this season.

Gameweek 1 does not buy. From Gameweek 2, a player with one or two prior
appearances and expected minutes of at least 45 may be bought. His
decision score is the published ``xp`` capped at 6. A player with three
or more appearances keeps the uncapped score. A player with none stays
out. No fixture-difficulty multiplier, no previous season, no ownership.

The climb starts from the crowd template and plays no chip. The bar was
locked before the totals: the four template climbs must sum to at least
7948, at least three seasons must match or beat their published climb,
and none may finish more than 30 behind it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.blank_context import clubs_by_gw, playable_gws
from src.models.crowd_opening_scores import GWS, load_openings, opening_state, path_result
from src.models.open_horizon import attach_opening_horizon
from src.models.ridge_multiseason import SEASONS
from src.models.season_climb_ft import EARLY_SCORE_CAP, load_vaastav_roster, run_ft_season

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SCORE_CSV = PROCESSED / "early_buy.csv"
SIGN_CSV = PROCESSED / "early_buy_signings.csv"
REPORT_PATH = REPORTS / "early_buy.md"

# Published template climbs. reports/crowd_opening_scores.md.
BASELINES: dict[str, float] = {
    "2022-23": 1735.0,
    "2023-24": 2105.0,
    "2024-25": 2086.0,
    "2025-26": 1988.0,
}
POOL_LIFT = 34.0
SEASONS_CLEAR = 3
COLLAPSE = 30.0
MIN_XMI = 45.0


def pool_bar() -> float:
    return float(sum(BASELINES.values()) + POOL_LIFT)


def cap_decision_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Cap ``score_xp`` where ``n_prior`` is 1 or 2. Other rows stay."""
    out = frame.copy()
    n_prior = pd.to_numeric(out["n_prior"], errors="coerce")
    score = pd.to_numeric(out["score_xp"], errors="coerce")
    capped = score.clip(upper=EARLY_SCORE_CAP)
    out["score_xp"] = score.where(~n_prior.between(1, 2), capped)
    return out


def buy_eligible(frame: pd.DataFrame) -> pd.Series:
    """True when this season's appearances and minutes clear the buy gate."""
    n_prior = pd.to_numeric(frame["n_prior"], errors="coerce")
    xmi = pd.to_numeric(frame["xmi"], errors="coerce").fillna(0.0)
    return (n_prior >= 1) & (xmi >= MIN_XMI)


def early_player_ids(pool: pd.DataFrame) -> set[str]:
    n_prior = pd.to_numeric(pool["n_prior"], errors="coerce")
    chosen = pool.loc[n_prior.between(1, 2), "player_id"]
    return set(chosen.astype(str))


def clip_horizon_steps(
    steps: dict[int, dict[str, float]],
    early_ids: set[str],
    cap: float = EARLY_SCORE_CAP,
) -> dict[int, dict[str, float]]:
    """A one-match share cannot price a later week above the cap."""
    clipped: dict[int, dict[str, float]] = {}
    for gw, scores in steps.items():
        row = dict(scores)
        for pid in early_ids:
            if pid in row:
                row[pid] = min(float(row[pid]), float(cap))
        clipped[int(gw)] = row
    return clipped


def decision_kept(points: dict[str, float]) -> dict[str, Any]:
    """The three bars. A missing season is not a pass."""
    diffs: dict[str, float] = {}
    complete = True
    for season, baseline in BASELINES.items():
        value = points.get(season)
        if value is None or value != value:
            complete = False
            diffs[season] = float("nan")
        else:
            diffs[season] = float(value) - float(baseline)
    pooled = float(sum(v for v in points.values() if v == v)) if complete else float("nan")
    ahead = int(sum(1 for gap in diffs.values() if gap == gap and gap >= 0.0))
    collapsed = [season for season, gap in diffs.items() if gap == gap and gap < -COLLAPSE]
    kept = bool(
        complete
        and pooled >= pool_bar()
        and ahead >= SEASONS_CLEAR
        and not collapsed
    )
    return {
        "kept": kept,
        "complete": complete,
        "pooled": pooled,
        "bar": pool_bar(),
        "ahead": ahead,
        "collapsed": collapsed,
        "diffs": diffs,
    }


def _signings(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Players who joined the fifteen with one or two prior appearances."""
    held: set[str] = set()
    rows: list[dict[str, Any]] = []
    for week in trace:
        squad = week["squad"]
        ids = set(squad["player_id"].astype(str))
        if not held:
            held = ids
            continue
        arrived = ids - held
        held = ids
        if not arrived:
            continue
        part = squad.loc[squad["player_id"].astype(str).isin(arrived)].copy()
        n_prior = pd.to_numeric(part["n_prior"], errors="coerce")
        part = part.loc[n_prior.between(1, 2)]
        for player in part.itertuples(index=False):
            rows.append(
                {
                    "gw": int(week["gw"]),
                    "player_id": str(player.player_id),
                    "n_prior": int(player.n_prior),
                    "xmi": float(player.xmi),
                    "score_xp": float(player.score_xp),
                    "minutes": float(getattr(player, "minutes", 0) or 0),
                    "total_points": float(getattr(player, "total_points", 0) or 0),
                }
            )
    return rows


def _capped_horizon(feat: pd.DataFrame):
    base = attach_opening_horizon(feat)
    if base is None:
        return None

    def horizon_scores(gw: int, pool: pd.DataFrame, gws: list[int]):
        steps = base(gw, pool, gws)
        return clip_horizon_steps(steps, early_player_ids(pool))

    return horizon_scores


def run_season(season: str, code: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One template climb. The opening week does not transfer."""
    from src.models.ridge_multiseason import build_one_season

    feat = build_one_season(season, code, early_buy=True)
    roster = load_vaastav_roster(season)
    expected = playable_gws(GWS, clubs_by_gw(roster))
    feat["eligible"] = buy_eligible(feat)
    slug = season.replace("-", "_")
    openings = load_openings()
    block = openings.loc[(openings["season"] == slug) & (openings["squad"] == "template")]
    opening = opening_state(season, block)
    trace: list[dict[str, Any]] = []
    weekly = run_ft_season(
        feat,
        {"xp": "score_xp"},
        GWS,
        roster=roster,
        chips=None,
        opening=opening,
        freeze_horizon=False,
        horizon_scores=_capped_horizon(feat),
        trace=trace,
    )
    result = path_result(weekly, expected, hold=False)
    result["season"] = season
    result["baseline"] = BASELINES[season]
    signings = _signings(trace)
    for row in signings:
        row["season"] = season
    return result, signings


def run(seasons: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    chosen = list(SEASONS if seasons is None else seasons)
    results: list[dict[str, Any]] = []
    signings: list[dict[str, Any]] = []
    for season, code in chosen:
        print(f"early buy {season}", flush=True)
        result, signed = run_season(season, code)
        results.append(result)
        signings.extend(signed)
        pd.DataFrame(results).to_csv(SCORE_CSV, index=False)
        pd.DataFrame(signings).to_csv(SIGN_CSV, index=False)
    points = {
        str(row["season"]): float(row["points"])
        for row in results
        if row.get("ok")
    }
    call = decision_kept(points)
    report = render_report(results, signings, call)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return {"results": results, "call": call, "signings": len(signings)}


def render_report(
    results: list[dict[str, Any]],
    signings: list[dict[str, Any]],
    call: dict[str, Any],
    *,
    review: str = "",
) -> str:
    lines = [
        "# Early buy",
        "",
        "From Gameweek 2, a player with one or two appearances this season "
        "and expected minutes of at least 45 may be bought. "
        f"His score is the published xp capped at {EARLY_SCORE_CAP:.0f}. "
        "Three or more appearances stay uncapped. "
        "No appearance stays unbuyable. "
        "Gameweek 1 does not transfer. "
        "There is no fixture-difficulty term, no previous season, and no ownership.",
        "",
        "Each season starts from the crowd template. The chip map is empty. "
        "The published template climbs are 1735, 2105, 2086, and 1988. "
        f"The pool must reach {pool_bar():.0f}. "
        f"At least {SEASONS_CLEAR} seasons must match or beat their climb. "
        f"None may finish more than {COLLAPSE:.0f} behind.",
        "",
    ]
    if review:
        lines.extend([review.strip(), ""])
    else:
        lines.extend(["The diagnostic review of these totals is still open.", ""])
    lines.append("| Season | Published | Early buy | Gap | Transfers | Hits | Early signings |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    by_season: dict[str, list[dict[str, Any]]] = {}
    for row in signings:
        by_season.setdefault(str(row["season"]), []).append(row)
    for row in results:
        season = str(row["season"])
        points = row.get("points")
        gap = ""
        shown = "failure"
        if row.get("ok") and points == points:
            shown = f"{float(points):.0f}"
            gap = f"{float(points) - BASELINES[season]:+.0f}"
        lines.append(
            f"| {season.replace('-', '/')} | {BASELINES[season]:.0f} | {shown} | {gap} "
            f"| {int(row.get('transfers') or 0)} | {int(row.get('hits') or 0)} "
            f"| {len(by_season.get(season, []))} |"
        )
    lines.append("")
    if call.get("complete"):
        word = "kept" if call["kept"] else "not kept"
        lines.append(
            f"The pool is {call['pooled']:.0f} against {call['bar']:.0f}. "
            f"{call['ahead']} seasons are level or ahead. "
            f"The buy is {word}."
        )
    else:
        lines.append("The four seasons are not all scored, so the buy is not kept.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
