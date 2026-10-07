"""Data eligibility for stored contrasts. The score is not repaired.

A week is eligible when its sheet has a nonzero expected goal for someone
who played and an earlier closed season is on disk. 2022-23 fails the
second test for every week. The all-weeks intervals stay published.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.loso import (
    CONTRASTS,
    LOSO_BOOTSTRAP,
    LOSO_CONDITIONAL_FLOOR,
    LOSO_FLOOR,
    LOSO_SEED,
    PUBLISHED_MEANS,
    SEASONS,
    _load_column,
    leave_one_out,
    pool_stored,
    reading,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PREDICTIONS = ROOT / "data" / "predictions" / "2026-27" / "gw06"
ENGINE_CAPTURE = PREDICTIONS / "20261007T070450Z.csv"
OFFICIAL_CAPTURE = PREDICTIONS / "official_20261007T080406Z.csv"
SHADOW_CAPTURE = PREDICTIONS / "shadow_20261007T080406Z.csv"

REQUIRED = (
    "The claim between the engine and expected points is inconclusive.",
    "The eligibility rule is post-hoc. It was written after 2022-23 was seen, and it uses the missing expected goals and the missing prior season, not the realised points.",
    "The eligible pool is the primary closed-season result. It covers three seasons and about 100 weeks. The all-weeks pool is a sensitivity check because it contains missing expected goals stored as zeros.",
    "The identical integer sum of −477 in 2022–23 is an arithmetic equivalence of season totals: dynamic transfers gained exactly 267 points over the opening fifteen under both scores.",
    "In 2022-23 the shared +267 is the season sum of that construction only. The weekly gains agree on 1 of 33 weeks, the players agree on 2 of 33 weeks, and the same-score transfer gains sum to +230 and +267. In 2023-24 the same construction sums to +461 and +438, and the players agree on 2 of 34 weeks. The greedy contrasts are not rechecked.",
    "Week by week, the two 2022–23 series agree on only 1 of 33 gameweeks and represent distinct processes; the shared season total remains unexplained and is not repaired.",
    "Gameweeks 1–15 of 2022–23 contain all-zero expected goals in the raw source, contaminating expanding priors throughout that season; 2022–23 also lacks a prior closed season in the archive.",
    "The eligible-weeks dataset requires populated xG and an available prior archive season, excluding 2022–23 while retaining all evaluated weeks of 2023–24, 2024–25, and 2025–26.",
    "An eligible interval that excludes zero is not a win and does not promote `score_xp`.",
    "The decision capture is the same-stamp pair written inside the window from three hours to fifteen minutes before the deadline. An earlier file is not used, and a capture is not chosen after the two scores have been compared.",
    "The 7 October shadow joins score_xp from 07:04 UTC to ep_next from 08:04 UTC and is not a decision pair. The 08:04 bootstrap was not stored, so that engine score cannot be rebuilt.",
    "No further closed-season analysis is added. The live review is at gameweek 26, with one outside audit at that review.",
    "From gameweek 6 onward, live squad selections are wired to pre-deadline official `ep_next`, with `score_xp` logged strictly in shadow.",
    "Promotion of `score_xp` requires a paired live interval strictly above zero across at least 20 pre-deadline gameweeks; covering zero remains undetermined through gameweek 38.",
    "The paired live comparison keeps every player who has both scores, and the column that drives the transfer does not drop the other score.",
)
FORBIDDEN = (
    "is suggestive",
    "are a week-by-week identity",
    "dropped from the all-weeks",
    "proves that `score_xp` outperforms",
    "has been promoted over",
    "imputed using `score_xp`",
    "A repair was applied",
)


def prior_season_available(season: str) -> bool:
    """An earlier season is in the closed archive. The scorer does not read it."""
    return str(season) in SEASONS and str(season) != SEASONS[0]


def week_eligible(*, xg_populated: bool, prior_season: bool) -> bool:
    return bool(xg_populated) and bool(prior_season)


def xg_week_table() -> pd.DataFrame:
    """One row per closed-season gameweek. A missing sheet is not populated."""
    rows: list[dict[str, Any]] = []
    for season in SEASONS:
        path = CACHE / f"merged_gw_{season.replace('-', '_')}.csv"
        frame = pd.read_csv(path, usecols=lambda name: name in {"GW", "round", "minutes", "expected_goals"})
        gw_name = "GW" if "GW" in frame.columns else "round"
        frame["gw"] = pd.to_numeric(frame[gw_name], errors="coerce")
        frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0)
        frame["expected_goals"] = pd.to_numeric(frame["expected_goals"], errors="coerce")
        played = frame.loc[frame["minutes"] > 0]
        prior = prior_season_available(season)
        grouped = {int(gw): block for gw, block in played.groupby("gw")}
        for gw in range(1, 39):
            block = grouped.get(gw)
            populated = block is not None and bool((block["expected_goals"].fillna(0.0) > 0).any())
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "xg_populated": populated,
                    "prior_season": prior,
                    "eligible": week_eligible(xg_populated=populated, prior_season=prior),
                }
            )
    return pd.DataFrame(rows)


def filter_weeks(rows: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Keep stored contrast rows whose gameweek passes the same rule."""
    marked = rows.copy()
    marked["season"] = marked["season"].astype(str)
    marked["gw"] = pd.to_numeric(marked["gw"], errors="coerce").astype(int)
    keep = flags.loc[flags["eligible"], ["season", "gw"]]
    return marked.merge(keep, on=["season", "gw"], how="inner")


def _pts(value: float) -> str:
    if abs(float(value) - round(float(value))) < 1e-9:
        return f"{int(round(float(value))):+d}"
    return f"{float(value):+.2f}"


def _plain(value: float) -> str:
    if abs(float(value) - round(float(value))) < 1e-9:
        return str(int(round(float(value))))
    return f"{float(value):.2f}"


def _interval(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "not identified"
    return f"{float(row['mean']):+.4f} [{float(row['lo']):+.4f}, {float(row['hi']):+.4f}]"


def _replay(season: str, protocol: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    from src.eval.common_state import _names
    from src.eval.decision import replay_season
    from src.eval.honest_pool import build_season

    frame = build_season(season, protocol["season_codes"][season], protocol)
    xp = replay_season(frame, "score_xp", gw_start=5, gw_end=38)
    exp = replay_season(frame, "score_exp_points", gw_start=5, gw_end=38)
    return _names(frame), xp, exp


def _label(names: dict[str, str], pid: str) -> str:
    if not pid:
        return "roll"
    return names.get(pid, pid)


def transfer_gains(protocol: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly transfer gains and the players moved. Two seasons, not a new contrast."""
    from src.eval.gates import load_protocol

    protocol = protocol or load_protocol()
    opening = pd.read_csv(PROCESSED / "initial_squad_weeks.csv")
    opening["gw"] = opening["gw"].astype(int)
    names, xp, exp = _replay("2022-23", protocol)
    paired = _join_opening(xp, exp, opening, "2022-23")
    rows = _gain_rows("2022-23", names, xp, exp, opening)
    names, xp, exp = _replay("2023-24", protocol)
    rows.extend(_gain_rows("2023-24", names, xp, exp, opening))
    return paired, pd.DataFrame(rows)


def _join_opening(
    xp: dict[str, Any],
    exp: dict[str, Any],
    opening: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    diverging = pd.DataFrame(
        {
            "gw": [int(row["gw"]) for row in xp["weeks"]],
            "div_xp": [float(row["greedy"]) for row in xp["weeks"]],
            "div_exp": [float(row["greedy"]) for row in exp["weeks"]],
        }
    )
    held = opening.loc[
        (opening["season"].astype(str) == season)
        & (opening["deployment"].astype(str) == "score_exp_points"),
        ["gw", "xp", "exp"],
    ].copy()
    paired = held.merge(diverging, on="gw", how="outer")
    if paired.isna().any().any() or len(paired) != 33:
        raise RuntimeError("the 2022-23 path totals do not cover the same 33 weeks")
    return paired.rename(columns={"xp": "open_xp", "exp": "open_exp"})


def _gain_rows(
    season: str,
    names: dict[str, str],
    xp: dict[str, Any],
    exp: dict[str, Any],
    opening: pd.DataFrame,
) -> list[dict[str, Any]]:
    xp_weeks = {int(row["gw"]): row for row in xp["weeks"]}
    exp_weeks = {int(row["gw"]): row for row in exp["weeks"]}
    xp_moves = {int(row["gw"]): row for row in xp["transfers"]}
    exp_moves = {int(row["gw"]): row for row in exp["transfers"]}
    port = opening.loc[
        (opening["season"].astype(str) == season)
        & (opening["deployment"].astype(str) == "score_exp_points")
    ]
    stored = {int(row.gw): row for row in port.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for gw in sorted(set(xp_weeks) & set(exp_weeks)):
        left = xp_moves.get(gw)
        right = exp_moves.get(gw)
        xin = "" if left is None else str(left["player_in"])
        xout = "" if left is None else str(left["player_out"])
        ein = "" if right is None else str(right["player_in"])
        eout = "" if right is None else str(right["player_out"])
        held = stored.get(gw)
        rows.append(
            {
                "season": season,
                "gw": gw,
                "hold_xp": float(xp_weeks[gw]["greedy"] - xp_weeks[gw]["hold"]),
                "hold_exp": float(exp_weeks[gw]["greedy"] - exp_weeks[gw]["hold"]),
                "pub_xp": float(xp_weeks[gw]["greedy"] - float(held.xp)),
                "pub_exp": float(exp_weeks[gw]["greedy"] - float(held.exp)),
                "xp_out": _label(names, xout),
                "xp_in": _label(names, xin),
                "exp_out": _label(names, eout),
                "exp_in": _label(names, ein),
                "same_players": xin == ein and xout == eout,
            }
        )
    return rows


def _assert_gains(gains: pd.DataFrame) -> None:
    """The footnote numbers. A drift here is a different replay, not a new result."""
    first = gains.loc[gains["season"] == "2022-23"]
    second = gains.loc[gains["season"] == "2023-24"]
    if len(first) != 33 or int(first["same_players"].sum()) != 2:
        raise RuntimeError("the 2022-23 transfer players are not the checked replay")
    if int(np.isclose(first["pub_xp"], first["pub_exp"]).sum()) != 1:
        raise RuntimeError("the 2022-23 weekly construction gains moved")
    if not (
        np.isclose(first["hold_xp"].sum(), 230.0)
        and np.isclose(first["hold_exp"].sum(), 267.0)
        and np.isclose(first["pub_xp"].sum(), 267.0)
        and np.isclose(first["pub_exp"].sum(), 267.0)
    ):
        raise RuntimeError("the 2022-23 transfer-gain sums moved")
    if len(second) != 34 or int(second["same_players"].sum()) != 2:
        raise RuntimeError("the 2023-24 transfer players are not the checked replay")
    if np.isclose(second["pub_xp"].sum(), second["pub_exp"].sum()):
        raise RuntimeError("2023-24 shares the construction sum")
    if not (
        np.isclose(second["pub_xp"].sum(), 461.0) and np.isclose(second["pub_exp"].sum(), 438.0)
    ):
        raise RuntimeError("the 2023-24 construction sums moved")


def _check_identity(paired: pd.DataFrame) -> None:
    open_xp = float(paired["open_xp"].sum())
    open_exp = float(paired["open_exp"].sum())
    div_xp = float(paired["div_xp"].sum())
    div_exp = float(paired["div_exp"].sum())
    if not (
        np.isclose(open_xp, 1237.0)
        and np.isclose(open_exp, 1714.0)
        and np.isclose(div_xp, 1504.0)
        and np.isclose(div_exp, 1981.0)
        and np.isclose(open_xp - open_exp, -477.0)
        and np.isclose(div_xp - div_exp, -477.0)
        and np.isclose(div_xp - open_xp, 267.0)
        and np.isclose(div_exp - open_exp, 267.0)
    ):
        raise RuntimeError(
            f"path totals moved: open {open_xp:.0f}/{open_exp:.0f} "
            f"diverging {div_xp:.0f}/{div_exp:.0f}"
        )
    gap_equal = int(np.isclose(paired["open_xp"] - paired["open_exp"], paired["div_xp"] - paired["div_exp"]).sum())
    if gap_equal != 1:
        raise RuntimeError(f"the weekly gaps agree on {gap_equal} weeks")


def _season_means(rows: pd.DataFrame, column: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for season in SEASONS:
        block = rows.loc[rows["season"].astype(str) == season, column]
        out[season] = None if block.empty else float(block.mean())
    return out


def _gain_lines(gains: pd.DataFrame) -> list[str]:
    lines = [
        "## Transfer gains",
        "",
        "In 2022-23 the shared +267 is the season sum of that construction only. "
        "The weekly gains agree on 1 of 33 weeks, the players agree on 2 of 33 weeks, "
        "and the same-score transfer gains sum to +230 and +267. "
        "In 2023-24 the same construction sums to +461 and +438, and the players agree on 2 of 34 weeks. "
        "The greedy contrasts are not rechecked.",
        "",
        "Same-score gain is greedy points minus that score's own frozen hold. "
        "Published gain is the diverging squad minus the frozen opening whose XI is chosen by expected points. "
        "That second series is the one whose 2022-23 sums are both +267.",
        "",
        "| season | GW | same-score xp | same-score exp | published xp | published exp | xp out | xp in | exp out | exp in |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    ordered = gains.sort_values(["season", "gw"])
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.season} | {int(row.gw)} | {_pts(float(row.hold_xp))} | {_pts(float(row.hold_exp))} | "
            f"{_pts(float(row.pub_xp))} | {_pts(float(row.pub_exp))} | {row.xp_out} | {row.xp_in} | "
            f"{row.exp_out} | {row.exp_in} |"
        )
    lines.append("")
    return lines


def _lines(
    paired: pd.DataFrame,
    flags: pd.DataFrame,
    pooled: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    gains: pd.DataFrame,
) -> list[str]:
    lines = [
        "The claim between the engine and expected points is inconclusive.",
        "",
        "The eligibility rule is post-hoc. It was written after 2022-23 was seen, and it uses "
        "the missing expected goals and the missing prior season, not the realised points.",
        "",
        "The eligible pool is the primary closed-season result. It covers three seasons and about 100 weeks. "
        "The all-weeks pool is a sensitivity check because it contains missing expected goals stored as zeros.",
        "2022-23 stays in the all-weeks sensitivity. The eligible table drops that season and keeps the later three.",
        "",
        "## Raw weekly totals, 2022-23",
        "",
        "The identical integer sum of −477 in 2022–23 is an arithmetic equivalence of season totals: "
        "dynamic transfers gained exactly 267 points over the opening fifteen under both scores.",
        "",
        "Week by week, the two 2022–23 series agree on only 1 of 33 gameweeks and represent distinct processes; "
        "the shared season total remains unexplained and is not repaired.",
        "",
        "Opening points are the frozen fifteens with the XI chosen by expected points. "
        "Diverging points are each score's own squad after its own transfers, with its own XI. "
        "There is no week-by-week identity between them.",
        "",
        "| GW | open xp | open exp | div xp | div exp |",
        "|---:|---:|---:|---:|---:|",
    ]
    for record in paired.itertuples(index=False):
        lines.append(
            f"| {int(record.gw)} | {_plain(record.open_xp)} | {_plain(record.open_exp)} | "
            f"{_plain(record.div_xp)} | {_plain(record.div_exp)} |"
        )
    lines.extend(
        [
            "",
            f"Season sums: opening score_xp {_plain(paired.open_xp.sum())}, "
            f"opening expected points {_plain(paired.open_exp.sum())}, "
            f"diverging score_xp {_plain(paired.div_xp.sum())}, "
            f"diverging expected points {_plain(paired.div_exp.sum())}.",
            "Gameweeks 5–15 of the gaps sum to "
            f"{_pts(float((paired.open_xp - paired.open_exp).loc[paired.gw.between(5, 15)].sum()))} "
            "on the opening path and "
            f"{_pts(float((paired.div_xp - paired.div_exp).loc[paired.gw.between(5, 15)].sum()))} "
            "on the diverging path. Gameweeks 26–38 sum to "
            f"{_pts(float((paired.open_xp - paired.open_exp).loc[paired.gw.between(26, 38)].sum()))} "
            "and "
            f"{_pts(float((paired.div_xp - paired.div_exp).loc[paired.gw.between(26, 38)].sum()))}.",
            "",
        ]
    )
    lines.extend(_gain_lines(gains))
    lines.extend(
        [
            "## xG stored as zero",
            "",
            "Gameweeks 1–15 of 2022–23 contain all-zero expected goals in the raw source, "
            "contaminating expanding priors throughout that season; 2022–23 also lacks a prior "
            "closed season in the archive.",
            "",
            "`compute_xp` reads the shift-1 expanding mean of those fields. The ingest stores a "
            "missing value as 0. This rule does not change that line and does not turn a prior "
            "season on.",
            "",
            "| season | GW | xG populated | prior season on disk | eligible |",
            "|---|---:|---|---|---|",
        ]
    )
    focus = flags.loc[flags["season"] == "2022-23"]
    for record in focus.itertuples(index=False):
        lines.append(
            f"| 2022-23 | {int(record.gw)} | {bool(record.xg_populated)} | "
            f"{bool(record.prior_season)} | {bool(record.eligible)} |"
        )
    later = flags.loc[flags["season"] != "2022-23"]
    lines.extend(
        [
            "",
            "The eligible-weeks dataset requires populated xG and an available prior archive season, "
            "excluding 2022–23 while retaining all evaluated weeks of 2023–24, 2024–25, and 2025–26.",
            f"Populated weeks in those three seasons: {int(later['xg_populated'].sum())} of {len(later)}. "
            f"Eligible weeks among them: {int(later['eligible'].sum())}.",
            "",
            "## Primary result, then the all-weeks sensitivity",
            "",
            "An eligible interval that excludes zero is not a win and does not promote `score_xp`. "
            "Every eligible contrast of `score_xp` against expected points covers zero, so the "
            "closed-season data do not separate the two scores.",
            "",
            "The published unconditional three-week interval is −1.69 [−2.92, −0.52]. "
            "Its exclusion of zero does not survive without 2022-23. "
            "The claim between the engine and expected points is inconclusive.",
            "",
        ]
    )
    for row in pooled:
        lines.append(f"### {row['label']}")
        lines.append("")
        tag = " Strawman." if row["strawman"] else ""
        if row["conditional"]:
            tag = " Conditional: the 20-week floor does not apply."
        lines.append(f"All-weeks published mean {float(row['published_mean']):+.4f}.{tag}")
        means = " | ".join(
            f"{season} {'—' if value is None else f'{value:+.2f}'}"
            for season, value in row["all_means"].items()
        )
        lines.append(f"All-weeks season means: {means}.")
        eligible_means = " | ".join(
            f"{season} {'—' if value is None else f'{value:+.2f}'}"
            for season, value in row["eligible_means"].items()
        )
        lines.append(f"Eligible season means: {eligible_means}.")
        lines.append("")
        lines.append("| version | weeks | estimate | reading |")
        lines.append("|---|---:|---|---|")
        lines.append(
            f"| eligible pool | {row['eligible_weeks']} | {_interval(row['eligible'])} | {row['eligible_reading']} |"
        )
        lines.append("")
    lines.append("### Eligible leave-one-season-out")
    lines.append("")
    lines.append(
        "Each fold drops one season from the eligible rows and resamples the rest. "
        "B=1000 and the seed is 0. Holding out 2022-23 leaves the eligible pool, because that season has no eligible week."
    )
    lines.append("")
    current = ""
    for fold in folds:
        if fold["key"] != current:
            if current:
                lines.append("")
            current = str(fold["key"])
            lines.append(f"#### {fold['label']}")
            lines.append("")
            lines.append("| held out | weeks | estimate | reading |")
            lines.append("|---|---:|---|---|")
        weeks = sum(int(n) for n in (fold.get("n_gws") or {}).values())
        lines.append(
            f"| {fold['holdout']} | {weeks} | {_interval(fold)} | {fold['reading']} |"
        )
    lines.extend(
        [
            "",
            "From gameweek 6 onward, live squad selections are wired to pre-deadline official `ep_next`, "
            "with `score_xp` logged strictly in shadow.",
            "Promotion of `score_xp` requires a paired live interval strictly above zero across at least "
            "20 pre-deadline gameweeks; covering zero remains undetermined through gameweek 38.",
            "The paired live comparison keeps every player who has both scores, and the column that "
            "drives the transfer does not drop the other score.",
            "The decision capture is the same-stamp pair written inside the window from three hours "
            "to fifteen minutes before the deadline. An earlier file is not used, and a capture is "
            "not chosen after the two scores have been compared.",
            "The scheduled job writes `ep_next`. `score_xp` is regenerated in that same run when the "
            "live files are on the machine. A checkout without those files does not invent an engine score.",
            "The 7 October shadow joins score_xp from 07:04 UTC to ep_next from 08:04 UTC and is not "
            "a decision pair. The 08:04 bootstrap was not stored, so that engine score cannot be rebuilt.",
            "No further closed-season analysis is added. The live review is at gameweek 26, with one "
            "outside audit at that review.",
            "The shadow file `data/predictions/2026-27/gw06/shadow_20261007T080406Z.csv` is kept as "
            "that mismatched log. The two source captures were not overwritten.",
            "",
            "No winner is declared. `score_xp` is unchanged. No repair was applied to the stored expected goals.",
            "",
            "Gemini kept the transfer-gain footnote, the post-hoc label, and the same-stamp rule "
            "([eligibility](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
        ]
    )
    return lines


def run() -> None:
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    protocol = load_protocol()
    rule = protocol["eligibility"]
    if rule.get("winner") is not None or rule.get("repairs_score") or rule.get("replaces_published"):
        raise RuntimeError("the eligibility lock is not the diagnostic rule")
    if rule.get("combine") != "and" or rule.get("timing") != "post-hoc":
        raise RuntimeError("eligibility is not the conjunction")
    if rule.get("primary") != "eligible" or rule.get("all_weeks") != "sensitivity":
        raise RuntimeError("the eligible pool is not the primary result")
    live = protocol["live_primary"]
    if live.get("decision_capture") != "t1_same_stamp" or live.get("choose_after_seeing_scores"):
        raise RuntimeError("the decision capture is not locked")
    flags = xg_week_table()
    if int(flags.loc[flags["season"] == "2022-23", "eligible"].sum()) != 0:
        raise RuntimeError("2022-23 has an eligible week")
    later = flags.loc[flags["season"] != "2022-23"]
    if not bool(later["xg_populated"].all()):
        raise RuntimeError("a later season has an unpopulated xG week")
    paired, gains = transfer_gains(protocol)
    _check_identity(paired)
    _assert_gains(gains)
    n_boot = int(protocol["loso"]["bootstrap"])
    seed = int(protocol["loso"]["seed"])
    if n_boot != LOSO_BOOTSTRAP or seed != LOSO_SEED:
        raise RuntimeError("the draw is not the locked draw")
    pooled_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for spec in CONTRASTS:
        stored = _load_column(spec)
        eligible = filter_weeks(stored, flags)
        minimum = LOSO_CONDITIONAL_FLOOR if spec["conditional"] else LOSO_FLOOR
        summary = pool_stored(
            eligible,
            str(spec["column"]),
            minimum=minimum,
            n_boot=n_boot,
            seed=seed,
        )
        summary["reading"] = reading(summary)
        pooled_rows.append(
            {
                "key": str(spec["key"]),
                "label": str(spec["label"]),
                "strawman": bool(spec["strawman"]),
                "conditional": bool(spec["conditional"]),
                "published_mean": float(PUBLISHED_MEANS[str(spec["key"])]),
                "all_means": _season_means(stored, str(spec["column"])),
                "eligible_means": _season_means(eligible, str(spec["column"])),
                "eligible": summary,
                "eligible_weeks": sum(int(n) for n in (summary.get("n_gws") or {}).values()),
                "eligible_reading": reading(summary),
            }
        )
        for season in SEASONS:
            fold = leave_one_out(
                eligible,
                str(spec["column"]),
                holdout=season,
                minimum=minimum,
                n_boot=n_boot,
                seed=seed,
            )
            fold["key"] = str(spec["key"])
            fold["label"] = str(spec["label"])
            fold["reading"] = reading(fold)
            fold_rows.append(fold)
    lines = _lines(paired, flags, pooled_rows, fold_rows, gains)
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
    flags.to_csv(PROCESSED / "eligibility_weeks.csv", index=False)
    paired.to_csv(PROCESSED / "path_totals_2022_23.csv", index=False)
    gains.to_csv(PROCESSED / "transfer_gains.csv", index=False)
    pd.DataFrame(
        [
            {
                "contrast": row["key"],
                "mean": row["eligible"].get("mean"),
                "lo": row["eligible"].get("lo"),
                "hi": row["eligible"].get("hi"),
                "reading": row["eligible_reading"],
                "n_gws": json.dumps(row["eligible"].get("n_gws") or {}),
            }
            for row in pooled_rows
        ]
    ).to_csv(PROCESSED / "eligibility_pool.csv", index=False)
    pd.DataFrame(
        [
            {
                "contrast": fold["key"],
                "holdout": fold["holdout"],
                "mean": fold.get("mean"),
                "lo": fold.get("lo"),
                "hi": fold.get("hi"),
                "reading": fold["reading"],
                "n_gws": json.dumps(fold.get("n_gws") or {}),
            }
            for fold in fold_rows
        ]
    ).to_csv(PROCESSED / "eligibility_loso.csv", index=False)
    write_gated_report(
        REPORTS / "eligibility.md",
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
