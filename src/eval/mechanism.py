"""Post-hoc labels on the published 51 disagreement weeks.

Realised points do not assign a flag. The flags overlap. The minutes-versus-attack
slices partition those 51 weeks. Neither table replaces the unconditional contrast
or changes score_xp.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.disagreement import MIN_DISAGREE, conditional_interval

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FLAGS = (
    "minutes_to_exp",
    "attack_to_xp",
    "cheaper_xp",
    "cs_defcon_to_xp",
    "goals_assists_to_xp",
    "form_to_exp",
    "fixture_to_xp",
)
SLICES = ("both", "minutes_only", "attack_only", "neither")
SUM_FIELDS = ("xp_goals", "xp_assists", "xp_cs", "xp_defcon")
MEAN_FIELDS = ("score_exp_points", "lam_scored", "xmi", "attack_strength")
N_WEEKS = 51

REQUIRED = (
    "The disagreement classification is a post-hoc description formulated after observing summary transfer characteristics; it does not replace the unconditional contrast or conditional intervals.",
    "The four slices of the minutes-versus-attack partition are mutually exclusive and sum to the 51 disagreement weeks; overlapping category gaps must not be summed.",
    "No winner is declared between `score_xp` and `score_exp_points`, and `score_xp` is unchanged.",
    "A slice interval that excludes zero is not a win and does not change score_xp.",
)
FORBIDDEN = (
    "The eligible pool is the primary closed-season result.",
    "is suggestive",
    "prove that expected points is superior",
    "caused by pursuing attack strength",
    "justify adjusting the weight",
)


def _finite(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number):
        return float("nan")
    return number


def _above(left: object, right: object) -> bool:
    """Strict greater-than. A tie or a missing value is not an advantage."""
    first = _finite(left)
    second = _finite(right)
    if not np.isfinite(first) or not np.isfinite(second):
        return False
    return first > second


def _pid(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def slice_name(minutes_to_exp: bool, attack_to_xp: bool) -> str:
    if minutes_to_exp and attack_to_xp:
        return "both"
    if minutes_to_exp:
        return "minutes_only"
    if attack_to_xp:
        return "attack_only"
    return "neither"


def flag_row(attrs: dict[str, object]) -> dict[str, Any]:
    """Flags from pre-deadline attributes of the two buys. Realised points are ignored."""
    minutes = _above(attrs.get("exp_xmi"), attrs.get("xp_xmi"))
    attack = _above(attrs.get("xp_attack"), attrs.get("exp_attack"))
    flags = {
        "minutes_to_exp": minutes,
        "attack_to_xp": attack,
        "cheaper_xp": _above(attrs.get("exp_price"), attrs.get("xp_price")),
        "cs_defcon_to_xp": _above(attrs.get("xp_cs_defcon"), attrs.get("exp_cs_defcon")),
        "goals_assists_to_xp": _above(attrs.get("xp_goals_assists"), attrs.get("exp_goals_assists")),
        "form_to_exp": _above(attrs.get("exp_form"), attrs.get("xp_form")),
        "fixture_to_xp": _above(attrs.get("xp_lam"), attrs.get("exp_lam")),
    }
    flags["other"] = not any(bool(flags[name]) for name in FLAGS)
    flags["slice"] = slice_name(minutes, attack)
    return flags


def _strict_reduce(series: pd.Series, how: str) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.empty or bool(values.isna().any()):
        return float("nan")
    if how == "sum":
        return float(values.sum())
    return float(values.mean())


def _component_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    work = work.dropna(subset=["gw"])
    work["gw"] = work["gw"].astype(int)
    work["player_id"] = work["player_id"].astype(str)
    sums = [column for column in SUM_FIELDS if column in work.columns]
    means = [column for column in MEAN_FIELDS if column in work.columns]
    grouped = work.groupby(["gw", "player_id"], sort=False)
    pieces: list[pd.DataFrame] = []
    if sums:
        pieces.append(grouped[sums].agg(lambda series: _strict_reduce(series, "sum")))
    if means:
        pieces.append(grouped[means].agg(lambda series: _strict_reduce(series, "mean")))
    if not pieces:
        return pd.DataFrame()
    table = pieces[0]
    for extra in pieces[1:]:
        table = table.join(extra, how="outer")
    return table


def _cell(table: pd.DataFrame, gw: int, pid: str, column: str) -> float:
    if not pid or table.empty or (gw, pid) not in table.index or column not in table.columns:
        return float("nan")
    return _finite(table.at[(gw, pid), column])


def _pair_sum(table: pd.DataFrame, gw: int, pid: str, columns: tuple[str, ...]) -> float:
    values = [_cell(table, gw, pid, column) for column in columns]
    if any(not np.isfinite(value) for value in values):
        return float("nan")
    return float(sum(values))


def _attach(published: pd.DataFrame, table: pd.DataFrame, season: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block = published.loc[published["season"].astype(str) == season]
    for source in block.itertuples(index=False):
        gw = int(source.gw)
        xp_in = _pid(source.xp_in)
        exp_in = _pid(source.exp_in)
        xp_xmi = _finite(source.xp_xmi)
        exp_xmi = _finite(source.exp_xmi)
        xp_attack = _finite(source.xp_attack)
        exp_attack = _finite(source.exp_attack)
        looked_xmi = _cell(table, gw, xp_in, "xmi")
        looked_attack = _cell(table, gw, xp_in, "attack_strength")
        if np.isfinite(xp_xmi) and np.isfinite(looked_xmi) and abs(xp_xmi - looked_xmi) > 1e-6:
            raise RuntimeError(f"{season} GW{gw} xmi does not match the published disagreement row")
        if np.isfinite(xp_attack) and np.isfinite(looked_attack) and abs(xp_attack - looked_attack) > 1e-6:
            raise RuntimeError(f"{season} GW{gw} attack does not match the published disagreement row")
        attrs = {
            "xp_xmi": xp_xmi,
            "exp_xmi": exp_xmi,
            "xp_attack": xp_attack,
            "exp_attack": exp_attack,
            "xp_price": _finite(source.xp_price),
            "exp_price": _finite(source.exp_price),
            "xp_cs_defcon": _pair_sum(table, gw, xp_in, ("xp_cs", "xp_defcon")),
            "exp_cs_defcon": _pair_sum(table, gw, exp_in, ("xp_cs", "xp_defcon")),
            "xp_goals_assists": _pair_sum(table, gw, xp_in, ("xp_goals", "xp_assists")),
            "exp_goals_assists": _pair_sum(table, gw, exp_in, ("xp_goals", "xp_assists")),
            "xp_form": _cell(table, gw, xp_in, "score_exp_points"),
            "exp_form": _cell(table, gw, exp_in, "score_exp_points"),
            "xp_lam": _cell(table, gw, xp_in, "lam_scored"),
            "exp_lam": _cell(table, gw, exp_in, "lam_scored"),
            "r1_xp_minus_r1_exp": _finite(source.r1_xp_minus_r1_exp),
            "r3_xp_minus_r3_exp": _finite(source.r3_xp_minus_r3_exp),
        }
        flags = flag_row(attrs)
        rows.append({"season": season, "gw": gw, **attrs, **flags})
    return rows


def _band(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "undefined"
    if float(row["lo"]) > 0.0:
        return "the interval stays above zero"
    if float(row["hi"]) < 0.0:
        return "the interval stays below zero"
    return "the interval covers zero"


def _fmt(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "undefined"
    return f"{row['mean']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"


def _interval_row(frame: pd.DataFrame, n_boot: int, seed: int) -> dict[str, dict[str, Any]]:
    return {
        "r1": conditional_interval(frame, "r1_xp_minus_r1_exp", n_boot=n_boot, seed=seed, minimum=MIN_DISAGREE),
        "r3": conditional_interval(frame, "r3_xp_minus_r3_exp", n_boot=n_boot, seed=seed, minimum=MIN_DISAGREE),
    }


def _lines(labelled: pd.DataFrame, groups: dict[str, dict[str, Any]], later_weeks: int) -> list[str]:
    if int(labelled["slice"].isin(SLICES).sum()) != N_WEEKS:
        raise RuntimeError("a disagreement week has no slice")
    counts = labelled["slice"].value_counts()
    if int(sum(int(counts.get(name, 0)) for name in SLICES)) != N_WEEKS:
        raise RuntimeError("the minutes-versus-attack slices do not sum to 51")
    lines = [
        "# Disagreement mechanisms",
        "",
        "No change is made to score_xp.",
        "",
        "The disagreement classification is a post-hoc description formulated after observing "
        "summary transfer characteristics; it does not replace the unconditional contrast or "
        "conditional intervals.",
        "",
        "Each flag compares the two incoming players. A tie or a missing value does not fire. "
        "Realised points are not an input. Bonus points are not a class.",
        "",
        f"Disagreement weeks: {len(labelled)}. Weeks in 2023-24, 2024-25, and 2025-26: {later_weeks}. "
        "That count is not a separate contrast.",
        "",
        "The four slices of the minutes-versus-attack partition are mutually exclusive and sum to "
        "the 51 disagreement weeks; overlapping category gaps must not be summed.",
        "",
        "## Overlapping flags",
        "",
        "| flag | weeks | one-week gap | reading | three-week gap | reading |",
        "|---|---:|---|---|---|---|",
    ]
    for name in (*FLAGS, "other"):
        summary = groups[name]
        weeks = int(summary["weeks"])
        lines.append(
            f"| {name} | {weeks} | {_fmt(summary['r1'])} | {_band(summary['r1'])} | "
            f"{_fmt(summary['r3'])} | {_band(summary['r3'])} |"
        )
    lines.extend(
        [
            "",
            "## Minutes versus attack",
            "",
            "The table asks whether the realised gap sits on weeks where the expected-points buy "
            "has more expected minutes, or on weeks where the score_xp buy has more attack strength.",
            "",
            "| slice | weeks | one-week gap | reading | three-week gap | reading |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for name in SLICES:
        summary = groups[name]
        lines.append(
            f"| {name} | {int(summary['weeks'])} | {_fmt(summary['r1'])} | {_band(summary['r1'])} | "
            f"{_fmt(summary['r3'])} | {_band(summary['r3'])} |"
        )
    lines.extend(
        [
            "",
            "A slice interval that excludes zero is not a win and does not change score_xp.",
            "",
            "No winner is declared between `score_xp` and `score_exp_points`, and `score_xp` is unchanged.",
            "",
            "Gemini kept the flag definitions "
            "([mechanism](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
            "",
            "Bootstrap 1000, seed 0. A season with fewer than 5 weeks in a group is omitted. "
            "Fewer than two such seasons leaves the interval undefined.",
            "",
        ]
    )
    return lines


def run() -> None:
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    from src.eval.honest_pool import build_season

    protocol = load_protocol()
    rule = protocol["mechanism"]
    if rule.get("winner") is not None or rule.get("changes_score") or rule.get("uses_realised_points"):
        raise RuntimeError("the mechanism lock is not descriptive")
    if int(rule.get("n_weeks") or 0) != N_WEEKS or rule.get("timing") != "post-hoc":
        raise RuntimeError("the mechanism lock is not the 51 published weeks")
    n_boot = int(protocol["bootstrap"])
    seed = int(protocol["seed"])
    if n_boot != 1000 or seed != 0:
        raise RuntimeError("the draw is not the locked draw")
    published = pd.read_csv(PROCESSED / "disagreement_weeks.csv")
    if len(published) != N_WEEKS:
        raise RuntimeError("the published disagreement file is not 51 weeks")
    labelled_rows: list[dict[str, Any]] = []
    codes = protocol["season_codes"]
    for season in protocol["closed_seasons"]:
        print(f"mechanism {season}", flush=True)
        frame = build_season(season, codes[season], protocol)
        labelled_rows.extend(_attach(published, _component_table(frame), season))
    labelled = pd.DataFrame(labelled_rows)
    if len(labelled) != N_WEEKS:
        raise RuntimeError("a published disagreement week was not labelled")
    if set(labelled["slice"]) - set(SLICES):
        raise RuntimeError("a slice name is outside the partition")
    groups: dict[str, dict[str, Any]] = {}
    for name in (*FLAGS, "other"):
        block = labelled.loc[labelled[name].astype(bool)]
        groups[name] = {"weeks": len(block), **_interval_row(block, n_boot, seed)}
    for name in SLICES:
        block = labelled.loc[labelled["slice"] == name]
        groups[name] = {"weeks": len(block), **_interval_row(block, n_boot, seed)}
    later_weeks = int((labelled["season"].astype(str) != "2022-23").sum())
    lines = _lines(labelled, groups, later_weeks)
    text = "\n".join(lines)
    for sentence in REQUIRED:
        if sentence not in text:
            raise RuntimeError("a required sentence is missing")
    for banned in FORBIDDEN:
        if banned in text:
            raise RuntimeError(f"the report contains a banned claim: {banned}")
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    key = "score_xp_minus_score_exp_points"
    audit = run_asof_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(PROCESSED / "disagreement_mechanism.csv", index=False)
    write_gated_report(
        REPORTS / "disagreement_mechanism.md",
        audit,
        {
            "seasons": list(protocol["closed_seasons"]),
            "min_gws": int(protocol["min_gws_per_season"]),
            "comparisons": {key: certified["comparisons"][key]},
        },
        lines,
    )


if __name__ == "__main__":
    run()
