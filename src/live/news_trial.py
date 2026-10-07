"""Gameweeks 1–5 played with the loose news tags.

A tag writes minutes only where one was already set. A missing tag, and an
ask, leave the published score untouched. A written zero removes the week's
score and the shot share, so a player who will not play is not priced off
goals. The published climb is not overwritten.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.models.xp_engine import CS_PTS

ROOT = Path(__file__).resolve().parents[2]
TAGS = ROOT / "data" / "processed" / "news_tags_gw15.csv"
OUT_CSV = ROOT / "data" / "processed" / "news_trial_gw15.csv"
OUT_REPORT = ROOT / "reports" / "news_trial_gw15.md"
GWS = (1, 2, 3, 4, 5)
BUY_GATE = 45.0


def load_writes(path: Path | None = None) -> dict[tuple[int, int], float]:
    """Tagged minutes, keyed by gameweek and element id. An ask is absent."""
    writes: dict[tuple[int, int], float] = {}
    with (path or TAGS).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("xmi") or "").strip()
            if not raw:
                continue
            key = (int(row["gw"]), int(row["player_id"]))
            if key in writes:
                raise RuntimeError(f"two tags for element {key[1]} in gameweek {key[0]}")
            writes[key] = float(raw)
    return writes


def apply_tag_scores(
    feat: pd.DataFrame,
    writes: Mapping[tuple[int, int], float],
    roster: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Copy the frame and rescore only the rows a tag writes.

    Untagged rows keep the published score. A tagged minutes value that
    matches the published minutes is left alone. A zero clears the score
    and both shot shares. A positive value rebuilds the minutes-gated
    parts and leaves the goal and assist shares in place.

    A 0-minute week is absent from the published frame, and the climber
    then carries the last score. A tag for that week is written onto a
    new row. The points and minutes on that row come from the roster.
    """
    out = feat.copy()
    changed = 0
    zeroed = 0
    partial = 0
    for idx in list(out.index):
        element = _element(out.at[idx, "player_id"])
        key = (int(out.at[idx, "gw"]), element)
        if key not in writes:
            continue
        if _write_existing(out, idx, float(writes[key])):
            changed += 1
            if float(writes[key]) <= 0.0:
                zeroed += 1
            else:
                partial += 1
    inserted = _append_missing(out, writes, roster)
    if inserted:
        attrs = dict(out.attrs)
        out = pd.concat([out, pd.DataFrame(inserted)], ignore_index=True)
        out.attrs = attrs
        for payload in inserted:
            changed += 1
            if float(payload["xmi"]) <= 0.0:
                zeroed += 1
            else:
                partial += 1
    return out, {"changed": changed, "zeroed": zeroed, "partial": partial, "inserted": len(inserted)}


def run() -> dict[str, Any]:
    """Play Gameweeks 1–5 twice. The second frame carries the tags."""
    from src.live.benchmark import build_frames
    from src.models.season_climb_ft import run_ft_season

    feat, roster, _info = build_frames()
    writes = load_writes()
    tagged, counts = apply_tag_scores(feat, writes, roster)
    tagged.attrs["early_scores"] = feat.attrs.get("early_scores")
    _unchanged(feat, tagged)
    published_trace: list[dict[str, Any]] = []
    tagged_trace: list[dict[str, Any]] = []
    published = run_ft_season(
        feat, {"xp": "score_xp"}, list(GWS), roster=roster, trace=published_trace
    )
    trial = run_ft_season(
        tagged, {"xp": "score_xp"}, list(GWS), roster=roster, trace=tagged_trace
    )
    if set(published["gw"].astype(int)) != set(GWS) or set(trial["gw"].astype(int)) != set(GWS):
        raise RuntimeError("a climb did not finish five weeks")
    result = {
        "published": published.to_dict("records"),
        "trial": trial.to_dict("records"),
        "published_rows": _squad_rows(published_trace),
        "trial_rows": _squad_rows(tagged_trace),
        "counts": counts,
        "sanchez": _keeper(published_trace, tagged_trace, 140),
    }
    _write(result)
    return result


def _write_existing(frame: pd.DataFrame, idx: object, xmi: float) -> bool:
    """Rescore one row. False when the tagged minutes match the published ones."""
    if _same(frame.at[idx, "xmi"], xmi):
        return False
    frame.at[idx, "xmi"] = xmi
    if xmi <= 0.0:
        frame.at[idx, "score_xp"] = 0.0
        frame.at[idx, "share_xG"] = 0.0
        frame.at[idx, "share_xA"] = 0.0
        frame.at[idx, "eligible"] = False
        return True
    frame.at[idx, "score_xp"] = _partial_score(frame.loc[idx], xmi)
    frame.at[idx, "eligible"] = bool(xmi >= BUY_GATE)
    return True


def _append_missing(
    frame: pd.DataFrame,
    writes: Mapping[tuple[int, int], float],
    roster: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """A tagged week with no feature row. The last earlier row supplies the shares."""
    if roster is None or roster.empty:
        return []
    present = {(int(gw), _element(pid)) for pid, gw in zip(frame["player_id"], frame["gw"], strict=False)}
    actuals: dict[tuple[int, int], Any] = {}
    for row in roster.itertuples(index=False):
        actuals[(int(row.gw), _element(row.player_id))] = row
    elements = frame["player_id"].map(_element)
    inserted: list[dict[str, Any]] = []
    for (gw, element), xmi in writes.items():
        if (gw, element) in present:
            continue
        earlier = frame.loc[(elements == element) & (pd.to_numeric(frame["gw"], errors="coerce") < gw)]
        if earlier.empty:
            continue
        payload = earlier.sort_values("gw").iloc[-1].to_dict()
        payload["gw"] = gw
        payload["xmi"] = float(xmi)
        actual = actuals.get((gw, element))
        if actual is not None:
            for field in ("total_points", "minutes", "team", "team_norm", "position", "value", "player_name"):
                if hasattr(actual, field):
                    payload[field] = getattr(actual, field)
        if float(xmi) <= 0.0:
            payload["score_xp"] = 0.0
            payload["share_xG"] = 0.0
            payload["share_xA"] = 0.0
            payload["eligible"] = False
        else:
            payload["score_xp"] = _partial_score(pd.Series(payload), float(xmi))
            payload["eligible"] = bool(float(xmi) >= BUY_GATE)
        inserted.append(payload)
    return inserted


def _partial_score(row: pd.Series, xmi: float) -> float:
    """Minutes-gated parts at the new minutes. Goals and assists stay."""
    p_play = min(1.0, max(0.0, xmi / 15.0))
    p60 = min(1.0, max(0.0, (xmi - 30.0) / 30.0)) if xmi >= 30.0 else 0.0
    position = str(row["position"])
    goals = float(row["xp_goals"])
    assists = float(row["xp_assists"])
    clean = p60 * float(row["p_cs_mkt"]) * float(CS_PTS.get(position, 0.0))
    defcon = 0.0
    if position in {"DEF", "MID", "FWD"}:
        defcon = p60 * float(row["exp_defcon_hit"]) * 2.0
    saves = p60 * (float(row["expected_saves"]) / 3.0) if position == "GKP" else 0.0
    bonus = 0.18 * goals + 0.12 * assists + 0.08 * clean
    conceded = 0.0
    if position in {"GKP", "DEF"}:
        conceded = p60 * (float(row["lam_conceded"]) / 2.0)
    cards = (xmi / 90.0) * 0.15
    level = float(row["fwd_level_add"]) if "fwd_level_add" in row.index else 0.0
    return p_play + p60 + goals + assists + clean + defcon + saves + bonus - conceded - cards + level


def _unchanged(published: pd.DataFrame, tagged: pd.DataFrame) -> None:
    """A row whose minutes were not written keeps its score."""
    left = published.loc[:, ["player_id", "gw", "xmi", "score_xp"]].copy()
    right = tagged.loc[:, ["player_id", "gw", "xmi", "score_xp"]].copy()
    left["player_id"] = left["player_id"].astype(str)
    right["player_id"] = right["player_id"].astype(str)
    merged = left.merge(right, on=["player_id", "gw"], suffixes=("_published", "_tagged"))
    same = (merged["xmi_published"] - merged["xmi_tagged"]).abs() < 1e-9
    if not bool(same.any()):
        return
    gap = (merged.loc[same, "score_xp_published"] - merged.loc[same, "score_xp_tagged"]).abs().max()
    if float(gap) > 1e-9:
        raise RuntimeError(f"an untagged score moved by {gap}")


def _squad_rows(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in trace:
        gw = int(step["gw"])
        named = set(step["xi"]["player_id"].astype(str))
        captain = str(step["captain_id"])
        for row in step["squad"].itertuples(index=False):
            pid = str(row.player_id)
            rows.append(
                {
                    "gw": gw,
                    "player_id": pid,
                    "element": _element(pid),
                    "name": str(row.player_name),
                    "position": str(row.position),
                    "team": str(row.team),
                    "lineup": "XI" if pid in named else "bench",
                    "xp": float(row.score_xp),
                    "points": float(row.total_points),
                    "minutes": float(row.minutes),
                    "captain": pid == captain,
                }
            )
    return rows


def _keeper(
    published: list[dict[str, Any]],
    tagged: list[dict[str, Any]],
    element: int,
) -> list[dict[str, Any]]:
    left = {int(row["gw"]): row for row in _squad_rows(published) if row["element"] == element}
    right = {int(row["gw"]): row for row in _squad_rows(tagged) if row["element"] == element}
    lines = []
    for gw in GWS:
        pub = left.get(gw)
        trial = right.get(gw)
        lines.append({"gw": gw, "published": pub, "trial": trial})
    return lines


def _write(result: Mapping[str, Any]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    published = result["published_rows"]
    trial = result["trial_rows"]
    fields = ["side", "gw", "player_id", "name", "position", "team", "lineup", "xp", "points", "minutes"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for side, rows in (("published", published), ("tags", trial)):
            for row in rows:
                writer.writerow({"side": side, **row})
    OUT_REPORT.write_text(_render(result), encoding="utf-8")


def _render(result: Mapping[str, Any]) -> str:
    published_weeks = {int(row["gw"]): row for row in result["published"]}
    trial_weeks = {int(row["gw"]): row for row in result["trial"]}
    published_total = sum(float(row["xi_points_cap"]) for row in result["published"])
    trial_total = sum(float(row["xi_points_cap"]) for row in result["trial"])
    counts = result["counts"]
    lines = [
        "# Gameweeks 1–5 with the loose news tags",
        "",
        "The climber is the published one: free transfers, an empty chip map, "
        "the opening horizon, Gameweek 1 solved from the buy pool. "
        "The tag writes minutes only where the note already set a number. "
        "An ask, and a week with no note, keep the published score. "
        "A written zero clears that week's score and the shot share. "
        "A doubtful flag keeps the chance times his old minutes and is not zeroed. "
        "This file does not replace the published squad.",
        "",
        (
            f"Rows rescored: {counts['changed']}. "
            f"Zeros: {counts['zeroed']}. "
            f"Partial minutes: {counts['partial']}. "
            f"Weeks inserted because the published frame had no row: {counts['inserted']}."
        ),
        "",
        f"Published total {published_total:.0f}. Tagged total {trial_total:.0f}.",
        "",
        "| GW | Published | Tagged | Published transfers | Tagged transfers |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    pub_rows = result["published_rows"]
    tag_rows = result["trial_rows"]
    for gw in GWS:
        left = published_weeks[gw]
        right = trial_weeks[gw]
        lines.append(
            f"| {gw} | {float(left['xi_points_cap']):.0f} | {float(right['xi_points_cap']):.0f} | "
            f"{int(left['n_transfers'])} | {int(right['n_transfers'])} |"
        )
    lines.extend(["", "## Where the fifteens differ", ""])
    differed = False
    for gw in GWS:
        lost, gained = _squad_diff(pub_rows, tag_rows, gw)
        if not lost and not gained:
            continue
        differed = True
        lines.append(
            f"Gameweek {gw}: out {', '.join(lost) if lost else 'nobody'}. "
            f"In {', '.join(gained) if gained else 'nobody'}."
        )
    if not differed:
        lines.append("The fifteen is the same in every week.")
    flips = _xi_flips(pub_rows, tag_rows)
    if flips:
        lines.append("")
        lines.append("The named eleven changes. The substitute was already in the squad.")
        for note in flips:
            lines.append(note)
    lines.extend(["", "## Sánchez", ""])
    lines.append("| GW | Published lineup | Published xp | Tagged lineup | Tagged xp | Played |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: |")
    for row in result["sanchez"]:
        left = row["published"]
        right = row["trial"]
        lines.append(
            f"| {row['gw']} | {_lineup(left)} | {_xp(left)} | {_lineup(right)} | {_xp(right)} | "
            f"{_played(left, right)} |"
        )
    lines.extend(
        [
            "",
            "Played minutes are the result after the deadline. They are not an input.",
            "",
            "The points stay level with the published climb. "
            "From Gameweek 3 the tag benches Sánchez and names Verbruggen. "
            "The automatic substitute had already been bringing Verbruggen on, so the week totals do not move. "
            "The benching is the lineup chosen before the deadline. "
            "Gameweek 2 is unchanged, because that deadline did not yet have the loan.",
            "",
            f"Sheet: `{OUT_CSV.relative_to(ROOT)}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _squad_diff(
    published: list[dict[str, Any]],
    tagged: list[dict[str, Any]],
    gw: int,
) -> tuple[list[str], list[str]]:
    left = {row["player_id"]: row["name"] for row in published if int(row["gw"]) == gw}
    right = {row["player_id"]: row["name"] for row in tagged if int(row["gw"]) == gw}
    lost = [left[pid] for pid in left if pid not in right]
    gained = [right[pid] for pid in right if pid not in left]
    return sorted(lost), sorted(gained)


def _xi_flips(published: list[dict[str, Any]], tagged: list[dict[str, Any]]) -> list[str]:
    left = {(int(row["gw"]), row["player_id"]): row for row in published}
    right = {(int(row["gw"]), row["player_id"]): row for row in tagged}
    notes: list[str] = []
    for key in sorted(left):
        other = right.get(key)
        if other is None or left[key]["lineup"] == other["lineup"]:
            continue
        row = left[key]
        notes.append(
            f"Gameweek {key[0]}: {row['name']} moves from {row['lineup']} to {other['lineup']}."
        )
    return notes


def _lineup(row: dict[str, Any] | None) -> str:
    if row is None:
        return "not owned"
    mark = " (C)" if row["captain"] else ""
    return f"{row['lineup']}{mark}"


def _xp(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    number = float(row["xp"])
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _played(left: dict[str, Any] | None, right: dict[str, Any] | None) -> str:
    row = left or right
    if row is None:
        return ""
    number = float(row["minutes"])
    if number == int(number):
        return str(int(number))
    return f"{number:.0f}"


def _element(value: object) -> int:
    return int(str(value).split(":")[-1])


def _same(published: object, xmi: float) -> bool:
    try:
        number = float(published)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if number != number:
        return False
    return abs(number - xmi) < 1e-9


if __name__ == "__main__":
    outcome = run()
    published = sum(float(row["xi_points_cap"]) for row in outcome["published"])
    trial = sum(float(row["xi_points_cap"]) for row in outcome["trial"])
    print(f"Published {published:.0f}. Tagged {trial:.0f}. Wrote {OUT_REPORT}")
