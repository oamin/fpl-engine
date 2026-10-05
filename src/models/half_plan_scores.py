"""Chip climb of the crowd Gameweek 1 fifteens.

The empty-chip totals stay in ``data/processed/crowd_opening_scores.csv``.
This run reads them and writes a side report. ``plan_half`` is solved again
at each deadline, before the transfer. 2024/25 is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.live.half_plan import (
    SquadOutlook,
    WeekInputs,
    bench_week,
    half_end,
    plan_half,
)
from src.live.policy import FH_MARGIN, WC_MARGIN
from src.models.blank_context import clubs_by_gw
from src.models.crowd_openings import SQUAD_LABEL, SQUAD_NAMES
from src.models.crowd_opening_scores import (
    GWS,
    load_openings,
    opening_state,
    prepare_season,
)
from src.models.open_horizon import attach_opening_horizon
from src.models.season_climb import pick_xi
from src.models.season_climb_ft import (
    SquadState,
    rebuild_squad,
    run_ft_season,
)
from src.rules.fpl_2026 import FREE_TRANSFER_CHIPS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
BASELINE_CSV = PROCESSED / "crowd_opening_scores.csv"
SCORE_CSV = PROCESSED / "half_plan_scores.csv"
REPORT_PATH = REPORTS / "half_plan_scores.md"
PARTIAL_DIR = Path("/tmp") / "half_plan_scores"

# The coded half-season wallet. 2024/25 stays out until Assistant Manager exists.
SCORE_SEASONS = (
    ("2022-23", "2223"),
    ("2023-24", "2324"),
    ("2025-26", "2526"),
)
WALLET_LABEL = {
    "2022-23": "2026 wallet on this season",
    "2023-24": "2026 wallet on this season",
    "2025-26": "matched wallet",
}
_ZERO = SquadOutlook(0.0, 0.0, 0.0)
SCORE_COL = "score_xp"


def club_steps(current_gw: int, clubs: dict[int, set[str]]) -> list[int]:
    """The next three gameweeks in this half that have a club fixture.

    A week with no clubs is left out. It is not a step that later weeks copy.
    """
    end = half_end(int(current_gw))
    playing = [
        gw
        for gw in range(int(current_gw), end + 1)
        if clubs.get(int(gw))
    ]
    return playing[:3]


def spread_outlooks(
    current_gw: int,
    clubs: dict[int, set[str]],
    priced: dict[int, tuple[SquadOutlook, SquadOutlook, float]],
) -> list[WeekInputs]:
    """One row per gameweek through the end of the half.

    ``priced`` is the up-to-three club weeks, in order. A week with no clubs
    is zero. A later club week reuses the last of those priced weeks.
    """
    steps = club_steps(current_gw, clubs)
    if list(priced) != steps:
        raise RuntimeError(
            f"GW{int(current_gw)} priced {list(priced)} and the club steps are {steps}"
        )
    if not steps:
        raise RuntimeError(f"GW{int(current_gw)} has no priced step")
    source = steps[-1]
    rows: list[WeekInputs] = []
    for gw in range(int(current_gw), half_end(int(current_gw)) + 1):
        if not clubs.get(int(gw)):
            rows.append(WeekInputs(int(gw), _ZERO, _ZERO, 0.0))
            continue
        held, rebuilt, fh_xi = priced[gw] if gw in priced else priced[source]
        rows.append(WeekInputs(int(gw), held, rebuilt, float(fh_xi)))
    return rows


def squad_outlook(
    pool: pd.DataFrame,
    ids: set[str],
    scores: dict[str, float],
    score_col: str = SCORE_COL,
) -> SquadOutlook:
    """Eleven plus the normal captain double. ``cap_xp`` is one further copy.

    A missing score is zero. The bench is the other players in ``ids``.
    """
    wanted = {str(pid) for pid in ids}
    frame = pool.loc[pool["player_id"].astype(str).isin(wanted)].copy()
    frame = frame.drop_duplicates("player_id", keep="first")
    frame[score_col] = [
        float(scores.get(str(pid), 0.0)) for pid in frame["player_id"]
    ]
    xi, _form = pick_xi(frame, score_col)
    xi_values = pd.to_numeric(xi[score_col], errors="coerce").fillna(0.0)
    all_values = pd.to_numeric(frame[score_col], errors="coerce").fillna(0.0)
    xi_sum = float(xi_values.sum())
    cap = float(xi_values.max()) if len(xi_values) else 0.0
    return SquadOutlook(
        xi_xp=xi_sum + cap,
        bench_xp=float(all_values.sum()) - xi_sum,
        cap_xp=cap,
    )


def _xi_with_captain(frame: pd.DataFrame, score_col: str) -> float:
    xi, _form = pick_xi(frame, score_col)
    xi_values = pd.to_numeric(xi[score_col], errors="coerce").fillna(0.0)
    return float(xi_values.sum()) + float(xi_values.max())


def free_hit_points(
    pool: pd.DataFrame,
    scores: dict[str, float],
    score_col: str = SCORE_COL,
) -> float:
    """Best starting eleven on this step, including the captain double.

    Eligible rows are the buy pool, and there is no budget. The first
    gameweeks have no buy pool yet, because three appearances are required.
    Those weeks use the rows in hand.
    """
    frame = pool.drop_duplicates("player_id", keep="first").copy()
    frame[score_col] = [
        float(scores.get(str(pid), 0.0)) for pid in frame["player_id"]
    ]
    if "eligible" in frame.columns:
        eligible = frame.loc[frame["eligible"].astype(bool)]
        try:
            return _xi_with_captain(eligible, score_col)
        except RuntimeError:
            pass
    return _xi_with_captain(frame, score_col)


def week_inputs(
    current_gw: int,
    state: SquadState,
    pool: pd.DataFrame,
    clubs: dict[int, set[str]],
    step_scores: dict[int, dict[str, float]],
    score_col: str = SCORE_COL,
) -> list[WeekInputs]:
    """Price the half from one decision-week rebuild.

    Wildcard and Free Hit share that fifteen on the decision week. A later
    Free Hit is the best eleven on that step. The rebuild is not solved again
    inside this call.
    """
    steps = club_steps(current_gw, clubs)
    rebuilt = rebuild_squad(state, pool, score_col)
    priced: dict[int, tuple[SquadOutlook, SquadOutlook, float]] = {}
    for index, gw in enumerate(steps):
        if int(gw) not in step_scores:
            raise RuntimeError(f"GW{int(gw)} has no outlook scores")
        scores = {str(pid): float(value) for pid, value in step_scores[int(gw)].items()}
        held = squad_outlook(pool, state.ids(), scores, score_col)
        rebuilt_out = squad_outlook(pool, rebuilt.ids(), scores, score_col)
        if index == 0:
            fh_xi = float(rebuilt_out.xi_xp)
        else:
            fh_xi = free_hit_points(pool, scores, score_col)
        priced[int(gw)] = (held, rebuilt_out, fh_xi)
    return spread_outlooks(current_gw, clubs, priced)


def make_chip_policy(horizon_scores, clubs: dict[int, set[str]], played: dict[int, str], sink=None):
    """Close over the shared horizon and the chips this squad has already used.

    ``sink(gw, state, pool, weeks, plan)`` sees the decision and does not
    choose it. The default is no sink.
    """

    def policy(gw, state, pool, gws):
        del gws
        if horizon_scores is None:
            raise RuntimeError("no opening-price horizon for this season")
        steps = club_steps(int(gw), clubs)
        raw = horizon_scores(int(gw), pool, steps)
        step_scores = {int(key): dict(value) for key, value in raw.items()}
        weeks = week_inputs(int(gw), state, pool, clubs, step_scores)
        plan = plan_half(int(gw), weeks, played=played)
        if sink is not None:
            sink(int(gw), state, pool, weeks, plan)
        chip = plan.chip
        bench = None if chip in FREE_TRANSFER_CHIPS else bench_week(plan, int(gw))
        print(f"    GW{int(gw)} {chip or 'none'}", flush=True)
        if chip is not None:
            played[int(gw)] = str(chip)
        return chip, bench

    return policy


def load_baseline(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(BASELINE_CSV if path is None else path)
    frame["climb_ok"] = frame["climb_ok"].map(_as_bool)
    frame["climb_points"] = pd.to_numeric(frame["climb_points"], errors="coerce")
    frame["climb_transfers"] = pd.to_numeric(frame["climb_transfers"], errors="coerce")
    frame["climb_hits"] = pd.to_numeric(frame["climb_hits"], errors="coerce")
    frame["climb_weeks"] = pd.to_numeric(frame["climb_weeks"], errors="coerce")
    return frame


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _failed(error: str) -> dict[str, object]:
    return {
        "points": float("nan"),
        "transfers": 0,
        "hits": 0,
        "weeks": [],
        "chips": "",
        "ok": False,
        "error": error,
    }


def _baseline(table: pd.DataFrame, season: str, squad: str) -> pd.Series | None:
    hit = table.loc[(table["season"] == season) & (table["squad"] == squad)]
    if hit.empty:
        return None
    return hit.iloc[0]


def chip_result(
    weekly: pd.DataFrame,
    expected: list[int],
    stored_weeks: int | None,
) -> dict[str, object]:
    """Sum the published week score. A short run fails the squad."""
    if weekly is None or weekly.empty or "method" not in weekly.columns:
        return _failed("no weeks scored")
    part = weekly.loc[weekly["method"] == "xp_ft"].copy()
    if part.empty:
        return _failed("no weeks scored")
    weeks = sorted(int(gw) for gw in part["gw"].unique())
    reasons: list[str] = []
    if weeks != sorted(int(gw) for gw in expected):
        reasons.append(f"scored {len(weeks)} weeks, expected {len(expected)}")
    if stored_weeks is not None and len(weeks) != int(stored_weeks):
        reasons.append("chip weeks do not match the stored climb")
    chips: list[str] = []
    for row in part.sort_values("gw").itertuples():
        chip = getattr(row, "chip", None)
        if chip is None or str(chip) in {"", "None", "nan"}:
            continue
        chips.append(f"GW{int(row.gw)} {chip}")
    return {
        "points": float(part["xi_points_cap"].sum()),
        "transfers": int(part["n_transfers"].sum()),
        "hits": int(part["hits"].sum()),
        "weeks": weeks,
        "chips": "; ".join(chips),
        "ok": not reasons,
        "error": "; ".join(reasons),
    }


def summarise(
    season: str,
    squad: str,
    baseline: pd.Series | None,
    chip: dict[str, object],
) -> dict[str, object]:
    empty_ok = baseline is not None and _as_bool(baseline["climb_ok"])
    empty_points = float(baseline["climb_points"]) if empty_ok else float("nan")
    chip_ok = bool(chip["ok"])
    chip_points = float(chip["points"])
    both = empty_ok and chip_ok and chip_points == chip_points and empty_points == empty_points
    lift = chip_points - empty_points if both else float("nan")
    return {
        "season": season,
        "squad": squad,
        "wallet": WALLET_LABEL.get(season, ""),
        "empty_points": empty_points,
        "empty_transfers": int(baseline["climb_transfers"]) if empty_ok else 0,
        "empty_hits": int(baseline["climb_hits"]) if empty_ok else 0,
        "empty_weeks": int(baseline["climb_weeks"]) if empty_ok else 0,
        "empty_ok": empty_ok,
        "chip_points": chip_points,
        "chip_transfers": int(chip["transfers"]),
        "chip_hits": int(chip["hits"]),
        "chip_weeks": len(chip["weeks"]),
        "chip_ok": chip_ok,
        "chip_error": str(chip["error"]),
        "lift": lift,
        "chips": str(chip["chips"]),
    }


def score_squad(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    clubs: dict[int, set[str]],
    horizon_scores,
    baseline: pd.DataFrame,
    season: str,
    squad: str,
    rows: pd.DataFrame,
    expected: list[int],
) -> dict[str, object]:
    """One fifteen on the chip policy. The stored climb is the baseline."""
    stored = _baseline(baseline, season, squad)
    stored_weeks = int(stored["climb_weeks"]) if stored is not None and _as_bool(stored["climb_ok"]) else None
    try:
        opening = opening_state(season, rows)
    except ValueError as exc:
        return summarise(season, squad, stored, _failed(str(exc)))
    known = set(roster["player_id"].astype(str))
    missing = sorted(opening.ids() - known)
    if missing:
        return summarise(
            season,
            squad,
            stored,
            _failed("missing from the roster: " + ", ".join(missing)),
        )
    played: dict[int, str] = {}
    policy = make_chip_policy(horizon_scores, clubs, played)
    try:
        weekly = run_ft_season(
            feat,
            {"xp": SCORE_COL},
            GWS,
            roster=roster,
            chips=None,
            opening=opening,
            horizon_scores=horizon_scores,
            freeze_horizon=False,
            chip_policy=policy,
        )
    except Exception as exc:
        return summarise(
            season, squad, stored, _failed(f"{type(exc).__name__}: {exc}")
        )
    return summarise(season, squad, stored, chip_result(weekly, expected, stored_weeks))


def score_all(
    openings: pd.DataFrame | None = None,
    seasons: list[tuple[str, str]] | None = None,
    checkpoint: Path | None = None,
    baseline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Three seasons, four squads. One season is built once."""
    openings = load_openings() if openings is None else openings
    chosen = list(SCORE_SEASONS if seasons is None else seasons)
    allowed = {season for season, _code in SCORE_SEASONS}
    base = load_baseline() if baseline is None else baseline
    summaries: list[dict[str, object]] = []
    for season, code in chosen:
        if season not in allowed:
            raise ValueError(f"{season} is not in this side report")
        print(f"building {season}", flush=True)
        slug = season.replace("-", "_")
        try:
            feat, roster, expected = prepare_season(season, code)
            horizon = attach_opening_horizon(feat)
            clubs = clubs_by_gw(roster)
        except Exception as exc:
            failed = _failed(f"{type(exc).__name__}: {exc}")
            for squad in SQUAD_NAMES:
                stored = _baseline(base, season, squad)
                summaries.append(summarise(season, squad, stored, failed))
            _checkpoint(summaries, checkpoint)
            continue
        if horizon is None:
            failed = _failed("no opening-price horizon for this season")
            for squad in SQUAD_NAMES:
                stored = _baseline(base, season, squad)
                summaries.append(summarise(season, squad, stored, failed))
            _checkpoint(summaries, checkpoint)
            continue
        crowd_key = slug
        for squad in SQUAD_NAMES:
            print(f"  {squad}", flush=True)
            block = openings.loc[
                (openings["season"] == crowd_key) & (openings["squad"] == squad)
            ]
            summaries.append(
                score_squad(
                    feat, roster, clubs, horizon, base, season, squad, block, expected
                )
            )
            _checkpoint(summaries, checkpoint)
    return pd.DataFrame(summaries)


def _checkpoint(summaries: list[dict[str, object]], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(path, index=False)


def best_lifts(table: pd.DataFrame) -> dict[str, list[str]]:
    """Largest chip-minus-empty gap inside each season. A tie keeps every name."""
    winners: dict[str, list[str]] = {}
    if table.empty:
        return winners
    for season, block in table.groupby("season", sort=False):
        done = block.loc[
            block["chip_ok"].map(_as_bool)
            & block["empty_ok"].map(_as_bool)
            & pd.to_numeric(block["lift"], errors="coerce").notna()
        ]
        if done.empty:
            winners[str(season)] = []
            continue
        top = float(pd.to_numeric(done["lift"], errors="coerce").max())
        names = done.loc[
            pd.to_numeric(done["lift"], errors="coerce") == top, "squad"
        ].astype(str).tolist()
        winners[str(season)] = names
    return winners


def _points(value: object) -> str:
    number = float(value)  # type: ignore[arg-type]
    if number != number:
        return "failure"
    return f"{number:.0f}"


def _signed(value: object) -> str:
    number = float(value)  # type: ignore[arg-type]
    if number != number:
        return "failure"
    return f"{number:+.0f}"


def _squad_name(squad: str) -> str:
    return SQUAD_LABEL.get(squad, squad)


def render_report(table: pd.DataFrame, *, review: str = "", reading: str = "") -> str:
    """Side report. The best lift is named inside its own season."""
    lines = [
        "# Half-season chip scores",
        "",
        "Each crowd Gameweek 1 fifteen is climbed again with `plan_half` "
        "solved at every deadline before the transfer. The decision score is "
        "`score_xp`, one fixture. The empty-chip climb is the stored total in "
        "`data/processed/crowd_opening_scores.csv`. The lift is the chip climb "
        "minus that stored total, so a chip that changes the squad is part of the gap.",
        "",
        "The seasons are 2022/23, 2023/24, and 2025/26. "
        "2024/25 is absent until Assistant Manager is in the rules module. "
        "2022/23 and 2023/24 are scored with the 2026 half-season wallet: "
        "one Wildcard, Free Hit, Bench Boost, and Triple Captain in each half. "
        "2025/26 is the season that wallet matches. "
        f"Free Hit needs a lead of {FH_MARGIN:.0f} on the decision week. "
        f"Wildcard needs a lead of {WC_MARGIN:.0f} across the half, on the eleven. "
        "Bench Boost and Triple Captain play when this week is the best week left "
        "and the added points are above zero. Gameweek 1 cannot play Wildcard or Free Hit.",
        "",
        "The decision week prices Wildcard and Free Hit from one legal fifteen, "
        "bought with the bank and the sell prices. A later Free Hit is the best "
        "starting eleven on that step's scores. The first gameweeks have no buy "
        "pool, so that eleven is taken from the rows in hand. Weeks after the third priced step "
        "reuse that step. A week with no clubs scores zero and stays out of the "
        "copied step. Realized points are Vaastav `total_points`, with the captain, "
        "automatic substitutes, the bench on Bench Boost, the extra captain copy "
        "on Triple Captain, and the hit deductions.",
        "",
        "The published climb file is unchanged. These totals stay in this side report.",
        "",
    ]
    if review:
        lines.extend([review.strip(), ""])
    else:
        lines.extend(["The diagnostic review of these totals is still open.", ""])
    if table.empty:
        lines.append("No squads were scored.")
        return "\n".join(lines).rstrip() + "\n"
    winners = best_lifts(table)
    for season, block in table.groupby("season", sort=False):
        label = str(season).replace("-", "/")
        lines.append(f"## {label}")
        lines.append("")
        lines.append(WALLET_LABEL.get(str(season), ""))
        lines.append("")
        lines.append(
            "| Squad | Empty climb | Chip climb | Chip − empty | Chips |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in block.iterrows():
            empty_cell = _points(row["empty_points"]) if _as_bool(row["empty_ok"]) else "failure"
            chip_cell = _points(row["chip_points"]) if _as_bool(row["chip_ok"]) else "failure"
            lift_cell = (
                _signed(row["lift"])
                if _as_bool(row["empty_ok"]) and _as_bool(row["chip_ok"])
                else "failure"
            )
            played = str(row["chips"]).strip()
            if not _as_bool(row["chip_ok"]):
                chip_names = "failure"
            elif played and played.lower() not in {"nan", "none"}:
                chip_names = played
            else:
                chip_names = "none"
            lines.append(
                f"| {_squad_name(str(row['squad']))} | {empty_cell} | "
                f"{chip_cell} | {lift_cell} | {chip_names} |"
            )
        lines.append("")
        for _, row in block.iterrows():
            if not _as_bool(row["chip_ok"]) and str(row["chip_error"]):
                lines.append(
                    f"{_squad_name(str(row['squad']))} — {row['chip_error']}."
                )
        if any(
            not _as_bool(row["chip_ok"]) and str(row["chip_error"])
            for _, row in block.iterrows()
        ):
            lines.append("")
        names = winners.get(str(season), [])
        if not names:
            lines.append(f"Best chip lift in {label}: none. Every chip climb failed.")
        elif len(names) == 1:
            winner = block.loc[block["squad"] == names[0]].iloc[0]
            lines.append(
                f"Best chip lift in {label}: {_squad_name(names[0])} "
                f"at {_signed(winner['lift'])}."
            )
        else:
            top = float(block.loc[block["squad"].isin(names), "lift"].iloc[0])
            joined = " and ".join(_squad_name(name) for name in names)
            lines.append(
                f"Best chip lift in {label}: {joined}, tied at {_signed(top)}."
            )
        lines.append("")
    if reading:
        lines.extend([reading.strip(), ""])
    lines.extend(
        [
            "A double stays one fixture on the decision score and inside the half plan. "
            "A Free Hit after the decision week has no budget cap, so a later Free Hit "
            "can look stronger than the fifteen the climb is able to buy. "
            "Weeks past the three priced steps reuse the last step, so a double or a "
            "blank that has not been announced is absent.",
            "",
            "Season totals are in `data/processed/half_plan_scores.csv`.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    table: pd.DataFrame | None = None,
    *,
    review: str = "",
    reading: str = "",
) -> pd.DataFrame:
    if table is None:
        table = score_all()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(SCORE_CSV, index=False)
    REPORT_PATH.write_text(
        render_report(table, review=review, reading=reading), encoding="utf-8"
    )
    return table


def combine_partials(directory: Path | None = None) -> pd.DataFrame:
    """Read the three season files written under /tmp."""
    folder = PARTIAL_DIR if directory is None else directory
    frames = []
    for season, _code in SCORE_SEASONS:
        path = folder / f"{season.replace('-', '_')}.csv"
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _one_season(season: str) -> None:
    """Write one season under /tmp so the three seasons can run together."""
    code = dict(SCORE_SEASONS)[season]
    slug = season.replace("-", "_")
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    table = score_all(seasons=[(season, code)], checkpoint=PARTIAL_DIR / f"{slug}.csv")
    print(f"{season} squads {len(table)}", flush=True)


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    if len(sys.argv) == 2 and sys.argv[1] == "--combine":
        scored = write_outputs(combine_partials())
        print(f"wrote {len(scored)} squads")
    elif len(sys.argv) == 2:
        _one_season(sys.argv[1])
    else:
        scored = write_outputs()
        print(f"wrote {len(scored)} squads")
