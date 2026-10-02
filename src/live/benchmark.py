"""2026/27 Gameweek 1–5 published xp climb, compared with a stored entry.

The score is ``score_xp``. The chip map is empty. Free-transfer hold,
switch penalty, and horizon stay on the published defaults. Last season
is attached only as a prior: those rows use a gameweek key before 2026
and are dropped before the squad is chosen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingest.fpl_odds import (
    join_players_to_fixtures,
    load_football_data,
    load_player_logs,
    parse_fpl_date,
)
from src.live.entry import stamp_matchday_teams
from src.live.fpl_snapshot import ELEMENT
from src.models.ridge_multiseason import _attach_value_defcon
from src.models.season_climb_ft import run_ft_season
from src.models.xp_engine import (
    DEFCON_THRESH,
    MIN_HISTORY,
    MIN_MINUTES,
    add_market_pots,
    add_player_priors,
    compute_xp,
)
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
HISTORY_PATH = CACHE / "fpl_history_2026_27.jsonl"
LOG_CSV = CACHE / "player_gw_2026_27.csv"
BOOTSTRAP_PATH = ROOT / "data" / "live" / "bootstrap.json"
FIXTURES_PATH = ROOT / "data" / "live" / "fixtures.json"
ENTRY_PATH = ROOT / "data" / "entry" / "2632584.json"

SEASON = "2026-27"
PRIOR_SEASON = "2025-26"
FD_CODE = "2627"
GWS = (1, 2, 3, 4, 5)
PRIOR_GW_SHIFT = 100
PRIOR_COLS = (
    "xmi",
    "exp_points",
    "roll3_points",
    "exp_xG",
    "exp_xA",
    "exp_defcon_hit",
)


def player_key(element: int | str) -> str:
    return f"{SEASON}:{int(element)}"


def assign_prior_id(
    element: str, code: int | None, code_to_2026: dict[int, int]
) -> str:
    """Stable id. Opta ``code`` links seasons. A missing code stays in 2025/26."""
    if code is not None and code in code_to_2026:
        return player_key(code_to_2026[code])
    return f"{PRIOR_SEASON}:{element}"


def stamp_unmatched_priors(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace a debutant's first-week NaN fill with matched pre-season priors.

    Matched players already carry a shift-1 prior from last season. Their
    first 2026 row is the position mean used for everyone else. Later 2026
    rows are not in that mean.
    """
    out = frame.copy()
    matched = set(out.loc[out["gw"] < 1, "player_id"].astype(str))
    current = out.loc[out["gw"] >= 1].sort_values(
        ["player_id", "gw", "date"], kind="mergesort"
    )
    first = current.groupby("player_id", sort=False).head(1)
    entry = first.loc[first["player_id"].astype(str).isin(matched)]
    means = entry.groupby("position")[list(PRIOR_COLS)].mean()
    debut = first.loc[~first["player_id"].astype(str).isin(matched)]
    for col in PRIOR_COLS:
        mapped = debut["position"].map(means[col]) if col in means.columns else np.nan
        out.loc[debut.index, col] = mapped.to_numpy()
    out["share_xG"] = (
        out["exp_xG"] / out["exp_team_xg"].replace(0, np.nan)
    ).clip(0, 1).fillna(0.0)
    out["share_xA"] = (
        out["exp_xA"] / out["exp_team_xa"].replace(0, np.nan)
    ).clip(0, 1).fillna(0.0)
    return out, int(len(debut))


def _scoring_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["xG"] = pd.to_numeric(out["xG"], errors="coerce").fillna(0.0)
    out["xA"] = pd.to_numeric(out["xA"], errors="coerce").fillna(0.0)
    out["p_not_lose"] = pd.to_numeric(out["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        out["p_draw"], errors="coerce"
    )
    out["p_over"] = pd.to_numeric(out["p_over25"], errors="coerce")
    out["p_under"] = pd.to_numeric(out["p_under25"], errors="coerce")
    thr = out["position"].map(DEFCON_THRESH)
    raw = pd.to_numeric(out.get("defcon_raw", 0.0), errors="coerce").fillna(0.0)
    out["defcon_raw"] = raw
    out["defcon_hit"] = (
        out["position"].isin(DEFCON_THRESH)
        & (pd.to_numeric(out["minutes"], errors="coerce").fillna(0) >= MIN_MINUTES)
        & (raw >= thr.fillna(999))
    ).astype(float)
    return out


def _prior_frame() -> tuple[pd.DataFrame, dict[str, int]]:
    players = load_player_logs(season=PRIOR_SEASON)
    players = _attach_value_defcon(players, PRIOR_SEASON)
    fixtures = load_football_data(code="2526")
    joined, _, stats = join_players_to_fixtures(players, fixtures)
    raw = pd.read_csv(CACHE / "players_raw_2025_26.csv")
    code_by_element = dict(zip(raw["id"].astype(str), raw["code"].astype(int)))
    boot = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    code_to_2026 = {int(e["code"]): int(e["id"]) for e in boot["elements"]}
    joined = joined.copy()
    joined["code"] = joined["player_id"].astype(str).map(code_by_element)
    joined["player_id"] = [
        assign_prior_id(str(element), None if pd.isna(code) else int(code), code_to_2026)
        for element, code in zip(joined["player_id"], joined["code"], strict=True)
    ]
    joined["gw"] = pd.to_numeric(joined["gw"], errors="coerce").astype(int) - PRIOR_GW_SHIFT
    n_linked = int(joined["player_id"].astype(str).str.startswith(f"{SEASON}:").sum())
    info = {
        "prior_joined": int(stats["n_player_joined"]),
        "prior_rows_linked": n_linked,
        "prior_players_linked": int(
            joined.loc[
                joined["player_id"].astype(str).str.startswith(f"{SEASON}:"), "player_id"
            ].nunique()
        ),
    }
    return _scoring_columns(joined), info


def _live_tables() -> tuple[dict[int, dict], dict[int, dict], dict[int, str]]:
    boot = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    elements = {int(e["id"]): e for e in boot["elements"]}
    teams = {int(t["id"]): str(t["name"]) for t in boot["teams"]}
    by_id = {int(f["id"]): f for f in fixtures}
    return elements, by_id, teams


def load_2026_logs() -> pd.DataFrame:
    """Player-gameweek rows. The CSV is the club they played for that week."""
    if LOG_CSV.exists():
        out = pd.read_csv(LOG_CSV)
        out.attrs["missing_fixture"] = 0
        return out
    elements, fixtures, teams = _live_tables()
    boot = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    shorts = {int(t["id"]): str(t["short_name"]) for t in boot["teams"]}
    rows: list[dict[str, Any]] = []
    missing_fixture = 0
    with HISTORY_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            blob = json.loads(line)
            element = int(blob["element"])
            person = elements.get(element)
            if person is None:
                continue
            pos = ELEMENT.get(int(person["element_type"]))
            if pos is None:
                continue
            for item in blob["history"]:
                gw = int(item["round"])
                if gw not in GWS:
                    continue
                fixture = fixtures.get(int(item["fixture"]))
                kickoff = item.get("kickoff_time") or ""
                if fixture is None or not kickoff:
                    missing_fixture += 1
                    continue
                home = bool(item.get("was_home"))
                team_id = int(fixture["team_h"] if home else fixture["team_a"])
                team = teams.get(team_id, "")
                if not team:
                    missing_fixture += 1
                    continue
                rows.append(
                    {
                        "date": parse_fpl_date(str(kickoff)),
                        "gw": gw,
                        "player_id": str(element),
                        "player_name": str(person["web_name"]),
                        "team": team,
                        "team_short": shorts[team_id],
                        "team_norm": norm_team(team),
                        "position": pos,
                        "is_home": 1 if home else 0,
                        "minutes": float(item.get("minutes") or 0),
                        "total_points": float(item.get("total_points") or 0),
                        "goals": float(item.get("goals_scored") or 0),
                        "assists": float(item.get("assists") or 0),
                        "clean_sheets": float(item.get("clean_sheets") or 0),
                        "goals_conceded": float(item.get("goals_conceded") or 0),
                        "saves": float(item.get("saves") or 0),
                        "bonus": float(item.get("bonus") or 0),
                        "bps": float(item.get("bps") or 0),
                        "xG": float(item.get("expected_goals") or 0),
                        "xA": float(item.get("expected_assists") or 0),
                        "defcon": float(item.get("defensive_contribution") or 0),
                        "yellow_cards": float(item.get("yellow_cards") or 0),
                        "red_cards": float(item.get("red_cards") or 0),
                        "starts": float(item.get("starts") or 0),
                        "value": float(item.get("value") or np.nan),
                        "defcon_raw": float(item.get("defensive_contribution") or 0),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No 2026/27 histories in {HISTORY_PATH}")
    out.attrs["missing_fixture"] = missing_fixture
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LOG_CSV, index=False)
    return out


def matchday_clubs(logs: pd.DataFrame) -> dict[tuple[int, int], str]:
    """``(element id, gw)`` to the short club name for that fixture."""
    clubs: dict[tuple[int, int], str] = {}
    if "team_short" not in logs.columns:
        return clubs
    for row in logs.itertuples(index=False):
        short = getattr(row, "team_short")
        if isinstance(short, str) and short and short != "nan":
            clubs[(int(row.player_id), int(row.gw))] = short
    return clubs


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return the climb features, the full roster, and join counts."""
    prior, info = _prior_frame()
    logs = load_2026_logs()
    roster = pd.DataFrame(
        {
            "player_id": [player_key(e) for e in logs["player_id"]],
            "gw": logs["gw"].astype(int),
            "player_name": logs["player_name"],
            "position": logs["position"],
            "team": logs["team"],
            "team_norm": logs["team_norm"],
            "value": pd.to_numeric(logs["value"], errors="coerce").fillna(50).astype(int),
            "total_points": logs["total_points"],
            "minutes": logs["minutes"],
        }
    ).drop_duplicates(["player_id", "gw"], keep="first")

    played = logs.loc[logs["minutes"] > 0].copy()
    fixtures = load_football_data(code=FD_CODE)
    joined, _, stats = join_players_to_fixtures(played, fixtures)
    joined = _scoring_columns(joined)
    joined["player_id"] = [player_key(e) for e in joined["player_id"]]
    joined["gw"] = pd.to_numeric(joined["gw"], errors="coerce").astype(int)
    if "value" not in joined.columns:
        joined["value"] = np.nan
    joined["value"] = pd.to_numeric(joined["value"], errors="coerce")
    joined["value"] = joined["value"].fillna(
        joined.groupby("position")["value"].transform("median")
    )

    combined = pd.concat([prior, joined], ignore_index=True, sort=False)
    scored = add_market_pots(combined)
    scored = add_player_priors(scored, fill_from=prior)
    scored, n_debut = stamp_unmatched_priors(scored)
    scored = compute_xp(scored)
    feat = scored.loc[scored["gw"].isin(GWS)].copy()
    feat = feat.loc[feat["n_prior"] >= MIN_HISTORY].copy()
    feat["score_xp"] = feat["xp"]
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    feat["player_id"] = feat["player_id"].astype(str)
    info.update(
        {
            "played_rows": int(len(played)),
            "joined_rows": int(stats["n_player_joined"]),
            "join_rate": float(stats["join_rate"]),
            "missing_fixture": int(logs.attrs.get("missing_fixture", 0)),
            "debut_rows_stamped": n_debut,
            "feat_rows": int(len(feat)),
            "odds_fixtures": int(stats["n_fixtures_odds"]),
        }
    )
    return feat, roster, info


def _names(frame: pd.DataFrame, captain: str, vice: str) -> list[str]:
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    view = frame.copy()
    view["_ord"] = view["position"].map(order).fillna(9)
    view = view.sort_values(["_ord", "name"], kind="mergesort")
    lines = []
    for rec in view.to_dict("records"):
        mark = ""
        pid = str(rec["player_id"])
        if pid == captain:
            mark = " (C)"
        elif pid == vice:
            mark = " (V)"
        lines.append(f"{rec['name']} {rec['position']} {rec['team']}{mark}")
    return lines


def _element(player_id: str) -> str:
    return str(player_id).split(":")[-1]


def run() -> dict[str, Any]:
    feat, roster, info = build_frames()
    gw1 = feat.loc[(feat["gw"] == 1) & feat["eligible"]]
    counts = gw1.groupby("position").size().to_dict()
    info["gw1_eligible"] = {pos: int(counts.get(pos, 0)) for pos in ("GKP", "DEF", "MID", "FWD")}
    info["gw1_eligible_n"] = int(len(gw1))
    if info["join_rate"] < 0.9:
        raise RuntimeError(f"Odds join rate {info['join_rate']:.3f} is too low to climb")
    for pos, need in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        if info["gw1_eligible"][pos] < need:
            raise RuntimeError(
                f"GW1 eligible {pos} {info['gw1_eligible'][pos]} < {need}. "
                "Not filled from later 2026/27 weeks."
            )

    trace: list[dict[str, Any]] = []
    weekly = run_ft_season(
        feat, {"xp": "score_xp"}, list(GWS), roster=roster, trace=trace
    )
    if weekly.empty or set(weekly["gw"]) != set(GWS):
        raise RuntimeError(
            f"Climb did not finish GW1–5 (rows={len(weekly)}, eligible GW1={info['gw1_eligible']})"
        )

    raw_entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    entry = stamp_matchday_teams(raw_entry, matchday_clubs(load_2026_logs()))
    if entry != raw_entry:
        ENTRY_PATH.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    picks = _pick_table(trace, roster, entry)
    model_picks = picks.loc[picks["side"] == "model"]
    for gw, week in model_picks.groupby("gw"):
        club_n = int(week.groupby("team").size().max())
        if club_n > 3:
            raise RuntimeError(f"GW{int(gw)} has {club_n} players from one club")
    summary = _summary(weekly, entry, roster, info)
    _write(weekly, picks, summary, info)
    return summary


def _pick_table(
    trace: list[dict[str, Any]], roster: pd.DataFrame, entry: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_gw = {int(g["gw"]): g for g in entry["gameweeks"]}
    for step in trace:
        gw = int(step["gw"])
        squad = step["squad"]
        xi_ids = set(step["xi"]["player_id"].astype(str))
        captain = str(step["captain_id"])
        vice = str(step["vice_id"])
        final_ids = set(step["final_xi"]["player_id"].astype(str))
        for row in squad.itertuples():
            pid = str(row.player_id)
            rows.append(
                {
                    "gw": gw,
                    "side": "model",
                    "role": "xi" if pid in xi_ids else "bench",
                    "in_scoring_xi": int(pid in final_ids),
                    "player_id": pid,
                    "name": row.player_name,
                    "position": row.position,
                    "team": row.team,
                    "is_captain": int(pid == captain),
                    "is_vice": int(pid == vice),
                    "xp": float(getattr(row, "score_xp", np.nan)),
                    "total_points": float(row.total_points),
                    "minutes": float(row.minutes),
                }
            )
        theirs = by_gw[gw]
        for role, block in (("xi", theirs["xi"]), ("bench", theirs["bench"])):
            for person in block:
                pid = player_key(person["id"])
                played = roster.loc[
                    (roster["gw"] == gw) & (roster["player_id"] == pid)
                ]
                points = float(played["total_points"].iloc[0]) if len(played) else np.nan
                minutes = float(played["minutes"].iloc[0]) if len(played) else np.nan
                rows.append(
                    {
                        "gw": gw,
                        "side": "ojaminFC",
                        "role": role,
                        "in_scoring_xi": int(role == "xi"),
                        "player_id": pid,
                        "name": person["name"],
                        "position": person["position"],
                        "team": person["team"],
                        "is_captain": int(person["name"] == theirs["captain"]),
                        "is_vice": int(person["name"] == theirs["vice"]),
                        "xp": np.nan,
                        "total_points": points,
                        "minutes": minutes,
                    }
                )
    return pd.DataFrame(rows)


def _summary(
    weekly: pd.DataFrame,
    entry: dict[str, Any],
    roster: pd.DataFrame,
    info: dict[str, Any],
) -> dict[str, Any]:
    model_total = float(weekly["xi_points_cap"].sum())
    theirs = float(entry["points"])
    gw1 = next(g for g in entry["gameweeks"] if int(g["gw"]) == 1)
    captain = next(p for p in gw1["xi"] if p["name"] == gw1["captain"])
    played = roster.loc[
        (roster["gw"] == 1) & (roster["player_id"] == player_key(captain["id"]))
    ]
    captain_pts = float(played["total_points"].iloc[0]) if len(played) else float("nan")
    # Triple Captain adds two copies. A normal captain adds one. The chip
    # premium versus the published rule is one copy of that captain.
    chip_premium = captain_pts if gw1.get("chip") == "triple_captain" else 0.0
    return {
        "model_points": model_total,
        "entry_points": theirs,
        "residual": model_total - theirs,
        "entry_without_tc_premium": theirs - chip_premium,
        "residual_without_tc_premium": model_total - (theirs - chip_premium),
        "haaland_gw1": captain_pts,
        "gw1_eligible": info["gw1_eligible"],
        "prior_players_linked": info["prior_players_linked"],
        "join_rate": info["join_rate"],
    }


def _write(
    weekly: pd.DataFrame,
    picks: pd.DataFrame,
    summary: dict[str, Any],
    info: dict[str, Any],
) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "live_benchmark_2026.csv", index=False)
    picks.to_csv(PROCESSED / "live_benchmark_2026_picks.csv", index=False)
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    by_gw = {int(g["gw"]): g for g in entry["gameweeks"]}
    lines = [
        "# 2026/27 xp climb against ojaminFC",
        "",
        "The climber is the published free-transfer path on `score_xp`. "
        "Hold margin 1.25, switch penalty 1.0, horizon 3, and an empty chip map. "
        "Nothing was retuned after these five weeks.",
        "",
        "Odds are football-data closing prices (`AvgCH`, `AvgC>2.5`), the same "
        "family the historical climbs use. They are not a pre-deadline book. "
        "No Odds API call was made.",
        "",
        "A different club in a later week is a transfer. The Opta code keeps the "
        "prior when only the spelling of the web name changed. The three-per-club "
        "cap uses the club they played for that week. A held squad that a transfer "
        "pushes over three from one club is not kept. A legal hold is unchanged.",
        "",
        "Gameweek 1 priors come from 2025/26. The link is the Opta `code` on the "
        f"FPL element, which matched {info['prior_players_linked']} players who "
        "appeared in both seasons. Those rows are stored with gameweek minus 100 "
        "so they sort before this season, then dropped. A player with no last-season "
        "row does not borrow a mean from later 2026/27 weeks. The buy pool is the "
        f"published cut: at least {MIN_HISTORY} prior appearances and expected minutes "
        "at least 45.",
        "",
        f"Odds join: {info['joined_rows']} of {info['played_rows']} appearances "
        f"({info['join_rate']:.3f}), {info['odds_fixtures']} fixtures. "
        "GW1 eligible pool: "
        + ", ".join(
            f"{pos} {info['gw1_eligible'][pos]}" for pos in ("GKP", "DEF", "MID", "FWD")
        )
        + ".",
        "",
        "## Points",
        "",
        f"- Model XI, captain doubled, hits removed: **{summary['model_points']:.0f}**",
        f"- ojaminFC, official gameweek total: **{summary['entry_points']:.0f}**",
        f"- Residual (model minus ojaminFC): **{summary['residual']:+.0f}**",
        "",
        "ojaminFC played Triple Captain in Gameweek 1 on Haaland. "
        f"Haaland's Gameweek 1 score was {summary['haaland_gw1']:.0f}. "
        "The published rule would have doubled that captain, so the chip added "
        f"one extra copy ({summary['haaland_gw1']:.0f} points). "
        f"Against that double-captain total the residual is "
        f"**{summary['residual_without_tc_premium']:+.0f}** "
        f"({summary['model_points']:.0f} versus {summary['entry_without_tc_premium']:.0f}).",
        "",
        "| GW | Model | ojaminFC | Gap | Formation | Transfers | Hits | Autosubs |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in weekly.sort_values("gw").itertuples():
        theirs = by_gw[int(row.gw)]["points"]
        gap = float(row.xi_points_cap) - float(theirs)
        lines.append(
            f"| {int(row.gw)} | {row.xi_points_cap:.0f} | {theirs} | {gap:+.0f} | "
            f"{row.formation} | {int(row.n_transfers)} | {int(row.hits)} | "
            f"{int(row.n_autosubs)} |"
        )
    lines += ["", "## Squads", ""]
    model = picks.loc[picks["side"] == "model"]
    previous: set[str] = set()
    for gw in GWS:
        week = model.loc[model["gw"] == gw]
        xi = week.loc[week["role"] == "xi"]
        bench = week.loc[week["role"] == "bench"]
        captain = str(week.loc[week["is_captain"] == 1, "player_id"].iloc[0])
        vice = str(week.loc[week["is_vice"] == 1, "player_id"].iloc[0])
        owned = set(week["player_id"].astype(str))
        bought = sorted(owned - previous)
        sold = sorted(previous - owned)
        theirs = by_gw[gw]
        their_ids = {player_key(p["id"]) for p in theirs["xi"] + theirs["bench"]}
        shared = len(owned & their_ids)
        lines.append(f"### GW{gw}")
        lines.append("")
        lines.append(
            "Selected XI: " + "; ".join(_names(xi, captain, vice)) + "."
        )
        scored = week.loc[week["in_scoring_xi"] == 1]
        if set(scored["player_id"]) != set(xi["player_id"]):
            lines.append(
                "Scoring XI after autosubs: "
                + "; ".join(_names(scored, captain, vice))
                + "."
            )
        lines.append(
            "Bench: " + "; ".join(_names(bench, captain, vice)) + "."
        )
        if previous:
            lines.append(
                "In: "
                + _id_names(week, bought)
                + ". Out: "
                + _id_names(picks.loc[(picks["side"] == "model") & (picks["gw"] == gw - 1)], sold)
                + "."
            )
            prev_week = model.loc[model["gw"] == gw - 1]
            moves = []
            for pid in sorted(owned & previous):
                before = prev_week.loc[prev_week["player_id"] == pid].iloc[0]
                after = week.loc[week["player_id"] == pid].iloc[0]
                if before["team"] != after["team"]:
                    moves.append(f"{after['name']} left {before['team']} for {after['team']}")
            if moves:
                lines.append(
                    "Already-owned transfer: "
                    + "; ".join(moves)
                    + ". The new club is the one that counts toward the cap."
                )
        lines.append(
            f"ojaminFC XI ({theirs['captain']} captain, {theirs['vice']} vice"
            + (", Triple Captain" if theirs["chip"] == "triple_captain" else "")
            + "): "
            + "; ".join(
                f"{p['name']} {p['position']} {p['team']}"
                + (" (C)" if p["name"] == theirs["captain"] else "")
                + (" (V)" if p["name"] == theirs["vice"] else "")
                for p in theirs["xi"]
            )
            + "."
        )
        lines.append(f"Squad overlap: {shared} of 15.")
        lines.append("")
        previous = owned
    lines += [
        "The model bank into a later week is not ojaminFC's bank. "
        "ojaminFC rolls 1 free transfer and £1.5m into Gameweek 6, with Wildcard, "
        "Free Hit, and Bench Boost still available. This run did not pick Gameweek 6.",
        "",
    ]
    (REPORTS / "live_benchmark_2026.md").write_text("\n".join(lines), encoding="utf-8")


def _id_names(week: pd.DataFrame, ids: list[str]) -> str:
    if not ids:
        return "none"
    names = []
    for pid in ids:
        hit = week.loc[week["player_id"] == pid]
        names.append(str(hit["name"].iloc[0]) if len(hit) else pid)
    return ", ".join(names)


def main() -> None:
    summary = run()
    print(
        f"model {summary['model_points']:.0f}  "
        f"ojaminFC {summary['entry_points']:.0f}  "
        f"residual {summary['residual']:+.0f}  "
        f"without TC premium {summary['residual_without_tc_premium']:+.0f}"
    )


if __name__ == "__main__":
    main()
