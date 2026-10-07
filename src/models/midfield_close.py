"""Five midfielders when the score gap is small.

The live window is the carried fifteens already scored in the shape trial.
A week whose named eleven is already five midfielders is left out: its
sacrifice and its gain are both zero by construction. The historical check
is the published fast eleven, Gameweeks 5–38, four seasons. That pool has
no bench, so the points are the eleven that was named. The two samples are
not added together.

The gates were locked before the slices were read. The decision band is a
sacrifice of 1.00 or less. The other cuts are printed and do not decide.
This run does not change the picker.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.reset_gap import PROCESSED, REPORTS
from src.models.season_climb import FORMATIONS, pick_xi
from src.models.shape_trial import EXCLUDED_LIMIT, FIVE_MID
from src.rules.fpl_2026 import MAX_PER_CLUB

FIVE_FORMS = {"3-5-2", "4-5-1"}
CUMULATIVE = (0.25, 0.50, 1.00, 1.50)
DECISION_CAP = 1.00
CONTROL_CAP = 1.50
WINDOW_N = 8
WINDOW_VETERAN_N = 4
SEASON_N = 20
SEASON_SKIP_MAX = 3
SPIKE_MEAN = 2.0
POOLED_CONTROL_N = 20
GW_LO = 5
GW_HI = 38
TOL = 1e-9
SOURCE = PROCESSED / "shape_trial_gw15.csv"
WINDOW_CSV = PROCESSED / "midfield_close_gw15.csv"
HISTORY_CSV = PROCESSED / "midfield_close_seasons.csv"
REPORT = REPORTS / "midfield_close.md"


def _feasible(row: dict[str, Any]) -> bool:
    value = row["feasible"]
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _num(value: Any) -> float:
    number = float(value)
    if number != number:
        raise RuntimeError("a feasible week has no sacrifice or gain")
    return number


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and the gain summaries. An empty band stays empty."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "median_sacrifice": None,
            "mean_gain": None,
            "median_gain": None,
            "n_positive": 0,
        }
    sacrifice = [_num(row["sacrifice"]) for row in rows]
    gain = [_num(row["gain"]) for row in rows]
    return {
        "n": n,
        "median_sacrifice": float(statistics.median(sacrifice)),
        "mean_gain": float(sum(gain) / n),
        "median_gain": float(statistics.median(gain)),
        "n_positive": sum(1 for points in gain if points > 0),
    }


def cumulative(rows: list[dict[str, Any]], cap: float) -> list[dict[str, Any]]:
    return [row for row in rows if _num(row["sacrifice"]) <= cap + TOL]


def control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _num(row["sacrifice"]) > CONTROL_CAP + TOL]


def disjoint_bin(
    rows: list[dict[str, Any]],
    lo: float | None,
    hi: float | None,
) -> list[dict[str, Any]]:
    """[0, hi] when lo is None. (lo, hi] otherwise. An open top has hi None."""
    chosen = []
    for row in rows:
        sacrifice = _num(row["sacrifice"])
        if lo is None:
            if sacrifice >= -TOL and (hi is None or sacrifice <= hi + TOL):
                chosen.append(row)
        elif sacrifice > lo + TOL and (hi is None or sacrifice <= hi + TOL):
            chosen.append(row)
    return chosen


def split_window(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate the 14 from the reference line, then drop the unusable weeks."""
    five = [row for row in records if row["alternate"] == "five midfielders"]
    cohort = [row for row in five if row["group"] != "reference"]
    reference = [row for row in five if row["group"] == "reference"]
    if len(cohort) != 70:
        raise RuntimeError(f"the window is 70 cohort rows, found {len(cohort)}")
    if len(reference) != 5:
        raise RuntimeError(f"the reference line is 5 rows, found {len(reference)}")
    groups = {str(row["group"]) for row in cohort}
    if groups != {"veteran", "rank"}:
        raise RuntimeError(f"unexpected groups: {sorted(groups)}")

    def _active(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        infeasible = [row for row in rows if not _feasible(row)]
        already = [
            row
            for row in rows
            if _feasible(row) and str(row["named_form"]) in FIVE_FORMS
        ]
        omega = [
            row
            for row in rows
            if _feasible(row) and str(row["named_form"]) not in FIVE_FORMS
        ]
        return {"infeasible": infeasible, "already": already, "omega": omega}

    return {"cohort": _active(cohort), "reference": _active(reference)}


def _group(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "14":
        return rows
    return [row for row in rows if row["group"] == group]


def band_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cumulative cuts, the control, and the disjoint bins. All descriptive except the call."""
    bins = [
        ("0 to 0.25", disjoint_bin(rows, None, 0.25)),
        ("0.25 to 0.50", disjoint_bin(rows, 0.25, 0.50)),
        ("0.50 to 1.00", disjoint_bin(rows, 0.50, 1.00)),
        ("1.00 to 1.50", disjoint_bin(rows, 1.00, 1.50)),
        ("above 1.50", disjoint_bin(rows, 1.50, None)),
    ]
    covered = [row for _label, block in bins for row in block]
    if len(covered) != len(rows):
        raise RuntimeError("a close-call week fell outside the bins")
    return {
        "cumulative": {f"{cap:.2f}": describe(cumulative(rows, cap)) for cap in CUMULATIVE},
        "control": describe(control_rows(rows)),
        "bins": [(label, describe(block)) for label, block in bins],
    }


def window_reading(
    *,
    n_infeasible: int,
    n_decision_14: int,
    n_decision_veterans: int,
    mean_decision_14: float | None,
    mean_decision_veterans: float | None,
    n_control_14: int,
    mean_control_14: float | None,
) -> dict[str, str]:
    """Present only when the locked gates all pass. The 0.25 cut is not an input."""
    if n_infeasible > EXCLUDED_LIMIT:
        return {"reading": "inconclusive", "control": "not reached"}
    sized = n_decision_14 >= WINDOW_N and n_decision_veterans >= WINDOW_VETERAN_N
    positive = (
        mean_decision_14 is not None
        and mean_decision_veterans is not None
        and mean_decision_14 > 0
        and mean_decision_veterans > 0
    )
    if not sized or not positive:
        return {"reading": "absent in this window", "control": "not reached"}
    if n_control_14 < WINDOW_N:
        return {"reading": "present in this window", "control": "too thin"}
    if mean_control_14 is None or mean_control_14 > mean_decision_14:
        return {"reading": "absent in this window", "control": "compared"}
    return {"reading": "present in this window", "control": "compared"}


def analyse_window(records: list[dict[str, Any]]) -> dict[str, Any]:
    parts = split_window(records)
    omega = parts["cohort"]["omega"]
    tables = {
        "14": band_table(omega),
        "veteran": band_table(_group(omega, "veteran")),
        "rank": band_table(_group(omega, "rank")),
        "reference": band_table(parts["reference"]["omega"]),
    }
    decision = tables["14"]["cumulative"]["1.00"]
    veterans = tables["veteran"]["cumulative"]["1.00"]
    call = window_reading(
        n_infeasible=len(parts["cohort"]["infeasible"]),
        n_decision_14=int(decision["n"]),
        n_decision_veterans=int(veterans["n"]),
        mean_decision_14=decision["mean_gain"],
        mean_decision_veterans=veterans["mean_gain"],
        n_control_14=int(tables["14"]["control"]["n"]),
        mean_control_14=tables["14"]["control"]["mean_gain"],
    )
    return {
        "n_infeasible": len(parts["cohort"]["infeasible"]),
        "n_already": len(parts["cohort"]["already"]),
        "n_omega": len(omega),
        "n_reference_already": len(parts["reference"]["already"]),
        "tables": tables,
        **call,
    }


def season_clears(*, n_skips: int, n_decision: int, mean_decision: float | None) -> bool:
    if n_skips > SEASON_SKIP_MAX:
        return False
    if n_decision < SEASON_N:
        return False
    return mean_decision is not None and mean_decision > 0


def is_spike(means: list[float | None]) -> bool:
    """One season above 2 and the other three at or below 0."""
    if len(means) != 4:
        raise RuntimeError("the spike check needs four seasons")
    hot = [mean for mean in means if mean is not None and mean > SPIKE_MEAN]
    cold = [mean for mean in means if mean is not None and mean <= 0]
    return len(hot) == 1 and len(cold) == 3


def history_reading(seasons: list[dict[str, Any]], pooled: dict[str, Any]) -> dict[str, str]:
    if len(seasons) != 4:
        raise RuntimeError("the historical reading needs four seasons")
    means = [row["decision"]["mean_gain"] for row in seasons]
    if is_spike(means):
        return {
            "reading": "retired",
            "control": "not reached",
            "note": "one season is above 2 and the other three are at or below 0",
        }
    n_clear = sum(1 for row in seasons if row["clears"])
    if n_clear < 3:
        return {
            "reading": "retired",
            "control": "not reached",
            "note": f"{n_clear} of 4 seasons clear the close-call gate",
        }
    control_n = int(pooled["control"]["n"])
    if control_n < POOLED_CONTROL_N:
        return {
            "reading": "kept",
            "control": "too thin",
            "note": "pooled control band too thin (n < 20); gate skipped",
        }
    decision_mean = pooled["decision"]["mean_gain"]
    control_mean = pooled["control"]["mean_gain"]
    if (
        decision_mean is None
        or control_mean is None
        or control_mean > decision_mean
    ):
        return {
            "reading": "retired",
            "control": "compared",
            "note": "weeks above 1.50 score more than the close calls",
        }
    return {
        "reading": "kept",
        "control": "compared",
        "note": "the control sits at or below the close-call gain",
    }


def analyse_history(rows: list[dict[str, Any]], seasons: list[str]) -> dict[str, Any]:
    """Rows are one published week each, already compared."""
    if seasons != sorted(set(seasons)) or len(seasons) != 4:
        raise RuntimeError("pass the four season names in order")
    built = []
    for season in seasons:
        block = [row for row in rows if row["season"] == season]
        omega = [row for row in block if row["status"] == "omega"]
        table = band_table(omega)
        decision = table["cumulative"]["1.00"]
        skips = sum(1 for row in block if row["status"] == "club")
        built.append(
            {
                "season": season,
                "n_weeks": len(block),
                "n_club": skips,
                "n_infeasible": sum(1 for row in block if row["status"] == "infeasible"),
                "n_already": sum(1 for row in block if row["status"] == "already"),
                "decision": decision,
                "control": table["control"],
                "table": table,
                "clears": season_clears(
                    n_skips=skips,
                    n_decision=int(decision["n"]),
                    mean_decision=decision["mean_gain"],
                ),
            }
        )
    pooled_omega = [row for row in rows if row["status"] == "omega"]
    pooled_table = band_table(pooled_omega)
    pooled = {
        "decision": pooled_table["cumulative"]["1.00"],
        "control": pooled_table["control"],
        "table": pooled_table,
    }
    call = history_reading(built, pooled)
    return {"seasons": built, "pooled": pooled, **call}


def batch_reading(window: str, history: str) -> str:
    """Both screens have to pass. Either one stopping is enough to leave the hypothesis."""
    if window == "present in this window" and history == "kept":
        return "kept"
    return "retired"


def _form_name(form: tuple[int, int, int]) -> str:
    return "-".join(str(part) for part in form)


def collapse_player(frame: pd.DataFrame) -> pd.DataFrame:
    """One player per week. A double keeps the earlier score and adds the points."""
    week = frame.copy()
    week["player_id"] = week["player_id"].astype(str)
    if "date" in week.columns:
        week = week.sort_values(["player_id", "date"], kind="mergesort")
    else:
        week = week.sort_values("player_id", kind="mergesort")
    points = week.groupby("player_id", sort=False)["total_points"].sum()
    base = week.drop_duplicates("player_id", keep="first").copy()
    base["total_points"] = base["player_id"].map(points).astype(float)
    return base.reset_index(drop=True)


def club_legal(xi: pd.DataFrame) -> bool:
    clubs = xi["team"].astype(str)
    if (clubs.str.strip() == "").any() or clubs.str.lower().eq("nan").any():
        raise RuntimeError("a selected player has no club")
    return int(clubs.value_counts().max()) <= MAX_PER_CLUB


def _starting(
    frame: pd.DataFrame, formations: list[tuple[int, int, int]]
) -> dict[str, Any] | None:
    try:
        xi, form = pick_xi(frame, "score_xp", formations=formations)
    except RuntimeError:
        return None
    return {
        "xi": xi,
        "form": tuple(int(part) for part in form),
        "xp": float(pd.to_numeric(xi["score_xp"], errors="coerce").fillna(0.0).sum()),
        "points": float(pd.to_numeric(xi["total_points"], errors="coerce").fillna(0.0).sum()),
    }


def compare_pool(frame: pd.DataFrame) -> dict[str, Any]:
    """Named eleven against five midfielders. No substitutes."""
    if "team" not in frame.columns:
        raise RuntimeError("the pool has no club")
    week = collapse_player(frame)
    named = _starting(week, list(FORMATIONS))
    alternate = _starting(week, list(FIVE_MID))
    if named is None or alternate is None:
        return {
            "status": "infeasible",
            "named_form": "" if named is None else _form_name(named["form"]),
            "alt_form": "" if alternate is None else _form_name(alternate["form"]),
            "sacrifice": None,
            "gain": None,
        }
    named_form = _form_name(named["form"])
    alt_form = _form_name(alternate["form"])
    if not club_legal(named["xi"]) or not club_legal(alternate["xi"]):
        return {
            "status": "club",
            "named_form": named_form,
            "alt_form": alt_form,
            "sacrifice": None,
            "gain": None,
        }
    if named_form in FIVE_FORMS:
        return {
            "status": "already",
            "named_form": named_form,
            "alt_form": alt_form,
            "sacrifice": 0.0,
            "gain": 0.0,
        }
    sacrifice = float(named["xp"] - alternate["xp"])
    if sacrifice < -1e-6:
        raise RuntimeError("the five-midfielder eleven outscored the maximum")
    return {
        "status": "omega",
        "named_form": named_form,
        "alt_form": alt_form,
        "sacrifice": sacrifice,
        "gain": float(alternate["points"] - named["points"]),
    }


def _fmt_gain(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-6:
        return f"{value:+.0f}"
    return f"{value:+.2f}"


def _fmt_sacrifice(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _stat_phrase(stats: dict[str, Any]) -> str:
    if stats["n"] == 0:
        return "no week"
    return (
        f"n {stats['n']}, median sacrifice {_fmt_sacrifice(stats['median_sacrifice'])}, "
        f"mean gain {_fmt_gain(stats['mean_gain'])}, "
        f"median gain {_fmt_gain(stats['median_gain'])}, "
        f"{stats['n_positive']} of {stats['n']} ahead"
    )


def _table_lines(table: dict[str, Any]) -> list[str]:
    lines = [
        "| Cut | Weeks | Median sacrifice | Mean gain | Median gain | Ahead |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cap in CUMULATIVE:
        stats = table["cumulative"][f"{cap:.2f}"]
        lines.append(_markdown_row(f"≤ {cap:.2f}", stats))
    lines.append(_markdown_row("> 1.50", table["control"]))
    for label, stats in table["bins"]:
        lines.append(_markdown_row(label, stats))
    return lines


def _markdown_row(label: str, stats: dict[str, Any]) -> str:
    if stats["n"] == 0:
        return f"| {label} | 0 |  |  |  |  |"
    return (
        f"| {label} | {stats['n']} | {_fmt_sacrifice(stats['median_sacrifice'])} | "
        f"{_fmt_gain(stats['mean_gain'])} | {_fmt_gain(stats['median_gain'])} | "
        f"{stats['n_positive']} of {stats['n']} |"
    )


def _window_sentence(result: dict[str, Any]) -> str:
    reading = result["reading"]
    if reading == "inconclusive":
        return (
            "More than 10 weeks cannot field five midfielders, so this window is inconclusive."
        )
    if reading == "present in this window":
        sentence = (
            "On this window the close call is present: a sacrifice of 1.00 or less "
            "has enough weeks, and the mean gain is positive for the 14 and for the veterans."
        )
    else:
        sentence = "On this window the close call is absent."
    if result["control"] == "too thin":
        sentence += " The weeks above 1.50 are fewer than 8, so that control was not applied."
    elif result["control"] == "compared" and reading == "present in this window":
        sentence += " The weeks above 1.50 do not score more than that close-call band."
    elif result["control"] == "compared":
        sentence += " Where the control was reached, it did not support the close call."
    return sentence


def _history_sentence(result: dict[str, Any]) -> str:
    n_clear = sum(1 for row in result["seasons"] if row["clears"])
    if result["reading"] == "kept":
        sentence = f"The historical check keeps the hypothesis. {n_clear} of 4 seasons clear."
    else:
        sentence = f"The historical check retires the hypothesis. {result['note']}."
        return sentence
    if result["control"] == "too thin":
        sentence += " The pooled weeks above 1.50 are fewer than 20, so that control was not applied."
    else:
        sentence += " The pooled weeks above 1.50 do not score more than the close calls."
    return sentence


def write_report(path: Path, window: dict[str, Any], history: dict[str, Any]) -> str:
    batch = batch_reading(window["reading"], history["reading"])
    if batch == "kept":
        ending = (
            "The hypothesis stays alive for a later squad test. "
            "The formation picker is unchanged."
        )
    else:
        ending = "The hypothesis is not kept. The formation picker is unchanged."
    lines = [
        "# Five midfielders in close calls",
        "",
        "The question is whether five midfielders score more actual points when their predicted score is within 1 of the named eleven. Always playing five midfielders is not the question. The picker, the score, the hold margin, and the switch penalty stay as they are.",
        "",
        "Sacrifice is the named score minus the five-midfielder score. Gain is the five-midfielder points minus the named points. A week that already named 3-5-2 or 4-5-1 is counted and then left out of every band. A tie on the score, broken toward an earlier formation, stays in the lowest band. The cuts at 0.25, 0.50, and 1.50 are printed. The gate is 1.00.",
        "",
        "The live window uses the carried fifteens and each shape's own substitutes. The four seasons use the published fast eleven from Gameweek 5 to 38: three prior appearances, and the joined log's players who played. There is no bench in that pool, so a blank stays on the sheet. A double keeps the earlier fixture's score and adds the points. A week with more than three from one club in either eleven is skipped. The two samples are not added together.",
        "",
        "The gates were locked before these slices were read. A live reading needs at least 8 close-call weeks on the 14 and 4 on the veterans, a positive mean gain on both, and, when at least 8 weeks sit above 1.50, a control mean that is not higher. More than 10 weeks that cannot field the shape make the window inconclusive. A season clears with at most 3 club skips, at least 20 close-call weeks, and a positive mean. The hypothesis needs the live reading and at least 3 of 4 seasons, and it is dropped if one season is above 2 while the other three are at or below 0. The pooled control uses the same idea once 20 weeks sit above 1.50.",
        "",
        "## Live window, Gameweeks 1–5",
        "",
        _window_sentence(window),
        "",
        (
            f"Already five midfielders: {window['n_already']} of 70. "
            f"Unable to field them: {window['n_infeasible']}. "
            f"Close calls: {window['n_omega']}."
        ),
        "",
    ]
    titles = (
        ("14", "The 14"),
        ("veteran", "Veterans"),
        ("rank", "Rank slots"),
        ("reference", "ojaminFC"),
    )
    for key, title in titles:
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(_table_lines(window["tables"][key]))
        lines.append("")
    lines.extend(
        [
            "## Fast eleven, 2022/23–2025/26",
            "",
            _history_sentence(history),
            "",
        ]
    )
    for season in history["seasons"]:
        lines.append(f"### {season['season']}")
        lines.append("")
        lines.append(
            f"Weeks {season['n_weeks']}. Club skips {season['n_club']}. "
            f"Unable to field five midfielders: {season['n_infeasible']}. "
            f"Already five midfielders: {season['n_already']}. "
            f"Close call at 1.00: {_stat_phrase(season['decision'])}. "
            f"{'Clears.' if season['clears'] else 'Does not clear.'}"
        )
        lines.append("")
        lines.extend(_table_lines(season["table"]))
        lines.append("")
    pooled = history["pooled"]
    lines.extend(
        [
            "### Four seasons pooled",
            "",
            f"Close call at 1.00: {_stat_phrase(pooled['decision'])}. "
            f"Above 1.50: {_stat_phrase(pooled['control'])}.",
            "",
        ]
    )
    lines.extend(_table_lines(pooled["table"]))
    lines.extend(["", "## Reading", "", ending, ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return batch


def _load_window(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def _history_rows() -> tuple[list[dict[str, Any]], list[str]]:
    from src.models.ridge_multiseason import SEASONS, build_one_season

    rows: list[dict[str, Any]] = []
    names = [season for season, _code in SEASONS]
    for season, code in SEASONS:
        print(f"close {season}", flush=True)
        feat = build_one_season(season, code)
        gws = sorted(
            int(gw)
            for gw in feat["gw"].unique()
            if GW_LO <= int(gw) <= GW_HI
        )
        for gw in gws:
            compared = compare_pool(feat.loc[feat["gw"] == gw])
            compared["season"] = season
            compared["gw"] = gw
            rows.append(compared)
    return rows, names


def run() -> dict[str, Any]:
    """Read the locked slices. Does not retune."""
    window = analyse_window(_load_window(SOURCE))
    history_rows, names = _history_rows()
    history = analyse_history(history_rows, names)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    five = source.loc[
        (source["alternate"] == "five midfielders") & (source["group"] != "reference")
    ]
    mask = five["feasible"].astype(bool) & ~five["named_form"].isin(FIVE_FORMS)
    five.loc[mask].to_csv(WINDOW_CSV, index=False)
    pd.DataFrame(history_rows).to_csv(HISTORY_CSV, index=False)
    batch = write_report(REPORT, window, history)
    return {
        "window": window["reading"],
        "window_control": window["control"],
        "history": history["reading"],
        "history_note": history["note"],
        "batch": batch,
        "n_already": window["n_already"],
        "n_omega": window["n_omega"],
    }


if __name__ == "__main__":
    print(run())
