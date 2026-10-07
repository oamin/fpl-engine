"""Side measurement of the stored chip climbs.

The chip rule, the margins, and ``data/processed/half_plan_scores.csv`` stay
as they are. A replay whose chip weeks, transfers, hits, or points differ
from that file fails before any split is written.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.live.half_plan import WeekInputs
from src.live.policy import WC_MARGIN
from src.models.crowd_opening_scores import (
    GWS,
    load_openings,
    opening_state,
    prepare_season,
)
from src.models.half_plan_scores import (
    SCORE_CSV,
    SCORE_SEASONS,
    WALLET_LABEL,
    chip_result,
    club_steps,
    make_chip_policy,
)
from src.models.open_horizon import attach_opening_horizon
from src.models.blank_context import clubs_by_gw
from src.models.crowd_openings import SQUAD_LABEL, SQUAD_NAMES
from src.models.season_climb import bank_squad_gw
from src.models.season_climb_ft import _fill_score, run_ft_season

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
EMPTY_WEEKS = PROCESSED / "crowd_opening_score_weeks.csv"
WILDCARD_CSV = PROCESSED / "chip_audit_wildcards.csv"
LIFT_CSV = PROCESSED / "chip_audit_lifts.csv"
CHURN_CSV = PROCESSED / "chip_audit_churn.csv"
WEEK_CSV = PROCESSED / "chip_audit_weeks.csv"
REPORT_PATH = REPORTS / "chip_audit.md"
PARTIAL_DIR = Path("/tmp") / "chip_audit"
SCORE_COL = "score_xp"
_CHIP_NAMES = {"wildcard", "free_hit", "bench_boost", "triple_captain"}


class AuditMismatch(RuntimeError):
    """The replay left the stored chip climb."""


def wildcard_margins(current_gw: int, weeks: list[WeekInputs], steps: list[int]) -> dict[str, float | int | bool]:
    """Split the wildcard eleven-gap into priced steps and the copied tail."""
    step_set = {int(gw) for gw in steps}
    gap_priced = 0.0
    gap_tail = 0.0
    for row in weeks:
        if row.rebuilt is None:
            raise AuditMismatch(f"GW{int(current_gw)} wildcard has no rebuild")
        gap = float(row.rebuilt.xi_xp) - float(row.held.xi_xp)
        if int(row.gw) in step_set:
            gap_priced += gap
        else:
            gap_tail += gap
    full = gap_priced + gap_tail
    return {
        "gap_priced": gap_priced,
        "gap_tail": gap_tail,
        "gap_all": full,
        "n_priced": len(step_set),
        "n_tail": len(weeks) - len(step_set),
        "clears_priced": gap_priced >= float(WC_MARGIN),
        "tail_share": (gap_tail / full) if full else float("nan"),
    }


def held_points(state, pool: pd.DataFrame, score_col: str = SCORE_COL) -> float:
    """Realized points of the squad in hand, with no transfer and no chip."""
    wanted = state.ids()
    squad = pool.loc[pool["player_id"].astype(str).isin(wanted)].copy()
    squad = squad.drop_duplicates("player_id", keep="first")
    squad[score_col] = _fill_score(squad, score_col)
    banked = bank_squad_gw(squad, score_col, use_autosubs=True)
    return float(banked["xi_points"]) + float(banked["cap_extra"])


def bench_award(squad: pd.DataFrame, final_xi: pd.DataFrame) -> float:
    """Points of the players left out of the final eleven."""
    final_ids = set(final_xi["player_id"].astype(str))
    total = 0.0
    frame = squad.drop_duplicates("player_id", keep="first")
    for row in frame.itertuples():
        if str(row.player_id) in final_ids:
            continue
        total += float(getattr(row, "total_points") or 0.0)
    return total


def _is_chip(value: object) -> bool:
    if value is None or (isinstance(value, float) and value != value):
        return False
    return str(value).strip() in _CHIP_NAMES


def partition_lift(
    chip_weeks: pd.DataFrame,
    empty_weeks: pd.DataFrame,
    mechanics: dict[int, float],
) -> dict[str, float]:
    """Split chip minus empty into the chip mechanic, the chip-week remainder, and later weeks.

    The three pieces add to the season lift. A chip week with no mechanic fails.
    """
    left = chip_weeks.loc[:, ["gw", "xi_points_cap", "chip"]].copy()
    right = empty_weeks.loc[:, ["gw", "xi_points_cap"]].copy()
    left["gw"] = left["gw"].astype(int)
    right["gw"] = right["gw"].astype(int)
    merged = left.merge(right, on="gw", suffixes=("_chip", "_empty"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise AuditMismatch("chip weeks and empty weeks do not cover the same gameweeks")
    active = 0.0
    path_gap = 0.0
    residual = 0.0
    for row in merged.itertuples():
        delta = float(row.xi_points_cap_chip) - float(row.xi_points_cap_empty)
        if _is_chip(row.chip):
            gw = int(row.gw)
            if gw not in mechanics:
                raise AuditMismatch(f"GW{gw} chip has no mechanic")
            mechanic = float(mechanics[gw])
            active += mechanic
            path_gap += delta - mechanic
        else:
            residual += delta
    return {
        "active": active,
        "path_gap": path_gap,
        "residual": residual,
        "lift": active + path_gap + residual,
    }


def churn_windows(
    chip_weeks: pd.DataFrame,
    empty_weeks: pd.DataFrame,
    wildcards: list[int],
) -> list[dict[str, object]]:
    """Transfers, hits, and bank after each wildcard, up to the next one."""
    ordered = sorted(int(gw) for gw in wildcards)
    rows: list[dict[str, object]] = []
    chip = chip_weeks.copy()
    empty = empty_weeks.copy()
    chip["gw"] = chip["gw"].astype(int)
    empty["gw"] = empty["gw"].astype(int)
    for index, start in enumerate(ordered):
        end = ordered[index + 1] if index + 1 < len(ordered) else None
        window = chip.loc[chip["gw"] > start]
        if end is not None:
            window = window.loc[window["gw"] < end]
        other = empty.loc[empty["gw"].isin(set(window["gw"].astype(int)))]
        if len(other) != len(window):
            raise AuditMismatch(f"GW{start} churn window is missing empty weeks")
        last_bank_chip = float(window["bank"].iloc[-1]) if len(window) else float("nan")
        last_bank_empty = float(other["bank"].iloc[-1]) if len(other) else float("nan")
        rows.append(
            {
                "wildcard_gw": start,
                "until_gw": end if end is not None else "",
                "chip_transfers": int(window["n_transfers"].sum()) if len(window) else 0,
                "empty_transfers": int(other["n_transfers"].sum()) if len(other) else 0,
                "chip_hits": int(window["hits"].sum()) if len(window) else 0,
                "empty_hits": int(other["hits"].sum()) if len(other) else 0,
                "chip_bank": last_bank_chip,
                "empty_bank": last_bank_empty,
            }
        )
    return rows


def _require_replay(
    weekly: pd.DataFrame,
    stored: pd.Series,
    expected: list[int],
) -> dict[str, object]:
    stored_weeks = int(stored["chip_weeks"])
    result = chip_result(weekly, expected, stored_weeks)
    reasons: list[str] = []
    if not result["ok"]:
        reasons.append(str(result["error"]))
    if abs(float(result["points"]) - float(stored["chip_points"])) > 1e-6:
        reasons.append(
            f"points {float(result['points']):.1f} against stored {float(stored['chip_points']):.1f}"
        )
    if int(result["transfers"]) != int(stored["chip_transfers"]):
        reasons.append(
            f"transfers {int(result['transfers'])} against stored {int(stored['chip_transfers'])}"
        )
    if int(result["hits"]) != int(stored["chip_hits"]):
        reasons.append(f"hits {int(result['hits'])} against stored {int(stored['chip_hits'])}")
    if str(result["chips"]) != str(stored["chips"]):
        reasons.append(f"chips {result['chips']!r} against stored {stored['chips']!r}")
    if reasons:
        raise AuditMismatch("; ".join(reasons))
    return result


def _mechanics(
    weekly: pd.DataFrame,
    trace: list[dict],
    held: dict[int, float],
) -> dict[int, float]:
    part = weekly.loc[weekly["method"] == "xp_ft"].sort_values("gw")
    if len(trace) != len(part):
        raise AuditMismatch("the score trace does not match the chip weeks")
    out: dict[int, float] = {}
    for row, item in zip(part.itertuples(), trace, strict=True):
        gw = int(row.gw)
        if int(item["gw"]) != gw:
            raise AuditMismatch(f"trace GW{int(item['gw'])} is not GW{gw}")
        chip = str(row.chip) if _is_chip(row.chip) else ""
        if chip == "bench_boost":
            out[gw] = bench_award(item["squad"], item["final_xi"])
        elif chip == "triple_captain":
            out[gw] = float(item["cap_extra"])
        elif chip in {"wildcard", "free_hit"}:
            if gw not in held:
                raise AuditMismatch(f"GW{gw} {chip} has no held score")
            out[gw] = float(row.xi_points_cap) - float(held[gw])
        elif chip:
            raise AuditMismatch(f"GW{gw} chip {chip} is not in the wallet")
    return out


def _empty_block(weeks: pd.DataFrame, season: str, squad: str) -> pd.DataFrame:
    block = weeks.loc[
        (weeks["season"] == season) & (weeks["squad"] == squad) & (weeks["path"] == "climb")
    ].copy()
    if block.empty:
        raise AuditMismatch(f"{season} {squad} has no empty climb weeks")
    return block


def replay_squad(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    clubs: dict[int, set[str]],
    horizon_scores,
    openings: pd.DataFrame,
    empty_weeks: pd.DataFrame,
    baseline: pd.DataFrame,
    season: str,
    squad: str,
    expected: list[int],
) -> dict[str, object]:
    """One crowd fifteen. A mismatch raises before the split is returned."""
    stored_rows = baseline.loc[(baseline["season"] == season) & (baseline["squad"] == squad)]
    if stored_rows.empty:
        raise AuditMismatch(f"{season} {squad} is missing from the chip file")
    stored = stored_rows.iloc[0]
    block = openings.loc[(openings["season"] == season.replace("-", "_")) & (openings["squad"] == squad)]
    opening = opening_state(season, block)
    played: dict[int, str] = {}
    margins: list[dict[str, object]] = []
    held: dict[int, float] = {}

    def sink(gw, state, pool, weeks, plan):
        chip = plan.chip
        if chip == "wildcard":
            steps = club_steps(int(gw), clubs)
            row = wildcard_margins(int(gw), list(weeks), steps)
            row.update({"season": season, "squad": squad, "gw": int(gw)})
            margins.append(row)
            held[int(gw)] = held_points(state, pool)
        elif chip == "free_hit":
            held[int(gw)] = held_points(state, pool)

    policy = make_chip_policy(horizon_scores, clubs, played, sink=sink)
    trace: list[dict] = []
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
        trace=trace,
    )
    result = _require_replay(weekly, stored, expected)
    empty = _empty_block(empty_weeks, season, squad)
    if abs(float(empty["xi_points_cap"].sum()) - float(stored["empty_points"])) > 1e-6:
        raise AuditMismatch(f"{season} {squad} empty weeks do not match the stored empty total")
    mechanics = _mechanics(weekly, trace, held)
    part = weekly.loc[weekly["method"] == "xp_ft"].copy()
    split = partition_lift(part, empty, mechanics)
    if abs(float(split["lift"]) - float(stored["lift"])) > 1e-6:
        raise AuditMismatch(
            f"{season} {squad} lift {float(split['lift']):.1f} against stored {float(stored['lift']):.1f}"
        )
    played_wc = [
        int(row.gw)
        for row in part.sort_values("gw").itertuples()
        if _is_chip(row.chip) and str(row.chip) == "wildcard"
    ]
    if sorted(int(row["gw"]) for row in margins) != played_wc:
        raise AuditMismatch(f"{season} {squad} wildcard log does not match the chip weeks")
    windows = churn_windows(part, empty, played_wc)
    for window in windows:
        window["season"] = season
        window["squad"] = squad
    week_rows = part.loc[:, ["gw", "chip", "xi_points_cap", "n_transfers", "hits", "bank"]].copy()
    week_rows["season"] = season
    week_rows["squad"] = squad
    empty_pts = empty.set_index(empty["gw"].astype(int))["xi_points_cap"]
    week_rows["empty_points"] = [float(empty_pts.loc[int(gw)]) for gw in week_rows["gw"]]
    week_rows["delta"] = week_rows["xi_points_cap"].astype(float) - week_rows["empty_points"]
    week_rows["mechanic"] = [mechanics.get(int(gw), 0.0) for gw in week_rows["gw"]]
    lift_row = {
        "season": season,
        "squad": squad,
        "wallet": WALLET_LABEL.get(season, ""),
        "chip_points": float(result["points"]),
        "empty_points": float(stored["empty_points"]),
        "lift": float(stored["lift"]),
        "active": float(split["active"]),
        "path_gap": float(split["path_gap"]),
        "residual": float(split["residual"]),
        "chips": str(result["chips"]),
    }
    return {"wildcards": margins, "lift": lift_row, "churn": windows, "weeks": week_rows}


def _season_bundle(season: str, code: str):
    feat, roster, expected = prepare_season(season, code)
    horizon = attach_opening_horizon(feat)
    if horizon is None:
        raise AuditMismatch(f"{season} has no opening-price horizon")
    clubs = clubs_by_gw(roster)
    return feat, roster, clubs, horizon, expected


def audit_season(
    season: str,
    code: str,
    openings: pd.DataFrame,
    empty_weeks: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, list]:
    feat, roster, clubs, horizon, expected = _season_bundle(season, code)
    wildcards: list[dict[str, object]] = []
    lifts: list[dict[str, object]] = []
    churn: list[dict[str, object]] = []
    weeks: list[pd.DataFrame] = []
    for squad in SQUAD_NAMES:
        print(f"  {squad}", flush=True)
        out = replay_squad(
            feat,
            roster,
            clubs,
            horizon,
            openings,
            empty_weeks,
            baseline,
            season,
            squad,
            expected,
        )
        wildcards.extend(out["wildcards"])
        lifts.append(out["lift"])
        churn.extend(out["churn"])
        weeks.append(out["weeks"])
    return {
        "wildcards": wildcards,
        "lifts": lifts,
        "churn": churn,
        "weeks": weeks,
    }


def _write_partial(season: str, bundle: dict[str, list]) -> None:
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    slug = season.replace("-", "_")
    pd.DataFrame(bundle["wildcards"]).to_csv(PARTIAL_DIR / f"{slug}_wildcards.csv", index=False)
    pd.DataFrame(bundle["lifts"]).to_csv(PARTIAL_DIR / f"{slug}_lifts.csv", index=False)
    pd.DataFrame(bundle["churn"]).to_csv(PARTIAL_DIR / f"{slug}_churn.csv", index=False)
    pd.concat(bundle["weeks"], ignore_index=True).to_csv(PARTIAL_DIR / f"{slug}_weeks.csv", index=False)


def load_partials(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    folder = PARTIAL_DIR if directory is None else directory
    frames = {name: [] for name in ("wildcards", "lifts", "churn", "weeks")}
    for season, _code in SCORE_SEASONS:
        slug = season.replace("-", "_")
        for name in frames:
            frames[name].append(pd.read_csv(folder / f"{slug}_{name}.csv"))
    return {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}


def _squad_name(squad: str) -> str:
    return SQUAD_LABEL.get(str(squad), str(squad))


def _num(value: object, digits: int = 1) -> str:
    number = float(value)  # type: ignore[arg-type]
    if number != number:
        return ""
    return f"{number:.{digits}f}"


def render_report(
    wildcards: pd.DataFrame,
    lifts: pd.DataFrame,
    churn: pd.DataFrame,
    *,
    review: str = "",
) -> str:
    """Side report. The stored chip file is not rewritten."""
    gw4 = wildcards.loc[wildcards["gw"].astype(int) == 4].copy()
    later = wildcards.loc[wildcards["gw"].astype(int) != 4].copy()
    n_clear = int(gw4["clears_priced"].map(lambda value: str(value).lower() in {"true", "1", "yes"}).sum()) if len(gw4) else 0
    lines = [
        "# Chip audit",
        "",
        "The twelve stored chip climbs were replayed. The chip rule, the margins "
        "of 12 and 16, and `data/processed/half_plan_scores.csv` were left as they "
        "are. A replay that changed a chip week, a transfer count, a hit count, or "
        "a season total would have stopped with no split.",
        "",
        "The wildcard gap is the decision-time eleven, rebuilt minus held. Priced "
        "steps are the next three club weeks. The tail is every later week in the "
        "half, including weeks that repeat the last priced step. The Gameweek 4 "
        "count is the twelve early wildcards, one on each climb. Later wildcards "
        "are listed on their own.",
        "",
        "The lift is the stored chip total minus the stored empty total. Active is "
        "the chip week only: bench points on Bench Boost, one extra captain copy "
        "on Triple Captain, and the played eleven minus the squad in hand on "
        "Wildcard and Free Hit. The path gap is the rest of those chip weeks "
        "against the empty climb. The residual is every week with no chip. "
        "The three pieces add to the lift. Hits the wildcard avoided are not "
        "simulated.",
        "",
        "Churn starts the week after each wildcard and stops before the next one. "
        "No new transfer search was run.",
        "",
    ]
    if review:
        lines.extend([review.strip(), ""])
    else:
        lines.extend(["The diagnostic review of these splits is still open.", ""])
    lines.append(f"Gameweek 4 wildcards that clear {WC_MARGIN:.0f} on priced steps alone: {n_clear} of {len(gw4)}.")
    lines.append("")
    lines.extend(_wildcard_table("Gameweek 4", gw4))
    lines.extend(_wildcard_table("Later wildcards", later))
    lines.extend(_lift_table(lifts))
    lines.extend(_churn_table(churn))
    lines.extend(
        [
            "The quote beside the stored chip file stays Next +34 (2171 to 2205) and Premium 2214.",
            "",
            "Wildcard rows are in `data/processed/chip_audit_wildcards.csv`. "
            "Lift rows are in `data/processed/chip_audit_lifts.csv`. "
            "Churn rows are in `data/processed/chip_audit_churn.csv`.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _wildcard_table(title: str, frame: pd.DataFrame) -> list[str]:
    lines = [f"## {title}", ""]
    if frame.empty:
        lines.extend(["None.", ""])
        return lines
    lines.append("| Season | Squad | GW | Priced gap | Tail gap | Full gap | Tail share | Priced clears 16 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    ordered = frame.sort_values(["season", "squad", "gw"])
    for row in ordered.itertuples():
        clears = str(row.clears_priced).lower() in {"true", "1", "yes"}
        lines.append(
            f"| {row.season} | {_squad_name(str(row.squad))} | {int(row.gw)} | "
            f"{_num(row.gap_priced, 1)} | {_num(row.gap_tail, 1)} | {_num(row.gap_all, 1)} | "
            f"{_num(row.tail_share, 2)} | {'yes' if clears else 'no'} |"
        )
    lines.append("")
    return lines


def _lift_table(frame: pd.DataFrame) -> list[str]:
    lines = ["## Lift", ""]
    lines.append("| Season | Squad | Lift | Active | Path gap | Residual |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in frame.sort_values(["season", "squad"]).itertuples():
        lines.append(
            f"| {row.season} | {_squad_name(str(row.squad))} | {_num(row.lift, 0)} | "
            f"{_num(row.active, 1)} | {_num(row.path_gap, 1)} | {_num(row.residual, 1)} |"
        )
    lines.append("")
    return lines


def _churn_table(frame: pd.DataFrame) -> list[str]:
    lines = ["## Churn after the wildcard", ""]
    lines.append(
        "| Season | Squad | Wildcard | Until | Chip transfers | Empty transfers | Chip hits | Empty hits | Chip bank | Empty bank |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    ordered = frame.sort_values(["season", "squad", "wildcard_gw"])
    for row in ordered.itertuples():
        until = "" if pd.isna(row.until_gw) or str(row.until_gw) in {"", "nan"} else str(int(float(row.until_gw)))
        lines.append(
            f"| {row.season} | {_squad_name(str(row.squad))} | {int(row.wildcard_gw)} | {until} | "
            f"{int(row.chip_transfers)} | {int(row.empty_transfers)} | {int(row.chip_hits)} | "
            f"{int(row.empty_hits)} | {_num(row.chip_bank, 0)} | {_num(row.empty_bank, 0)} |"
        )
    lines.append("")
    return lines


def write_outputs(bundle: dict[str, pd.DataFrame], *, review: str = "") -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    bundle["wildcards"].to_csv(WILDCARD_CSV, index=False)
    bundle["lifts"].to_csv(LIFT_CSV, index=False)
    bundle["churn"].to_csv(CHURN_CSV, index=False)
    bundle["weeks"].to_csv(WEEK_CSV, index=False)
    REPORT_PATH.write_text(
        render_report(bundle["wildcards"], bundle["lifts"], bundle["churn"], review=review),
        encoding="utf-8",
    )


def _one_season(season: str) -> None:
    code = dict(SCORE_SEASONS)[season]
    print(f"building {season}", flush=True)
    bundle = audit_season(
        season,
        code,
        load_openings(),
        pd.read_csv(EMPTY_WEEKS),
        pd.read_csv(SCORE_CSV),
    )
    _write_partial(season, bundle)
    print(f"{season} wildcards {len(bundle['wildcards'])}", flush=True)


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    if len(sys.argv) == 2 and sys.argv[1] == "--combine":
        write_outputs(load_partials())
        print("wrote chip audit")
    elif len(sys.argv) == 2:
        _one_season(sys.argv[1])
    else:
        for season, _code in SCORE_SEASONS:
            _one_season(season)
        write_outputs(load_partials())
        print("wrote chip audit")
