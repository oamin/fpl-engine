"""One live deadline for the squad already in the stored entry.

Purchase price is the Gameweek 1 value for a player still held, and the
transfer ``in_cost`` for a later buy. The site's selling price is what a
chip pays when the picks carry one. A minutes file is hashed and applied
as written, including a zero. A player left out of that file keeps his
last observed minutes and is labelled ``no_news``.

Gameweek 6 is not priced when its opening 1X2 is missing. A priced line
with a minutes file is scored by ``src.live.scorer`` and planned once.
This module does not invent a team rate, and it does not call the published climb.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.live.half_plan import HalfPlan, WeekInputs, bench_week, half_end, plan_half
from src.live.lines import TRIAL_META
from src.live.plan import live_xi
from src.live.policy import FH_MARGIN, WC_MARGIN
from src.models.forecast_xp import opening_pots_by_team_gw
from src.models.season_climb_ft import SquadState
from src.rules.fpl_2026 import FREE_TRANSFER_CHIPS, sell_price, squad_legal, xi_legal
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
ENTRY_PATH = ROOT / "data" / "entry" / "2632584.json"
LOG_PATH = ROOT / "data" / "cache" / "player_gw_2026_27.csv"
ODDS_PATH = ROOT / "data" / "cache" / "E0_2627.csv"
BOOTSTRAP_PATH = ROOT / "data" / "live" / "bootstrap.json"
FIXTURES_PATH = ROOT / "data" / "live" / "fixtures.json"
REPORT_PATH = ROOT / "reports" / "live_deadline_gw6.md"

DECISION_GW = 6
SEASON = "2026-27"
ENTRY_ID = 2632584

FORBIDDEN_NAMES = frozenset(
    {
        "crowd_opening_scores.csv",
        "half_plan_scores.csv",
        "season_climb_ft.csv",
        "live_benchmark_2026.md",
    }
)


class DeadlineError(ValueError):
    """The deadline log is missing a fact it is not allowed to invent."""


@dataclass(frozen=True)
class Holding:
    """One owned player at the deadline."""

    element: int
    key: str
    name: str
    position: str
    team: str
    purchase: int
    purchase_source: str
    current: int
    formula_sell: int
    selling: int
    selling_source: str

    @property
    def mismatch(self) -> bool:
        return self.selling_source == "api" and self.selling != self.formula_sell


@dataclass(frozen=True)
class DeadlineLog:
    """What was known before the deadline. ``chip`` is None when the run stops."""

    entry_id: int
    team_name: str
    gw: int
    deadline: str
    bank: int
    ft: int
    chips_played: tuple[tuple[int, str], ...]
    chips_left: tuple[str, ...]
    holdings: tuple[Holding, ...]
    state: SquadState
    minutes_file: str | None
    minutes_hash: str | None
    minutes: tuple[dict[str, Any], ...]
    line_status: str
    missing_clubs: tuple[str, ...]
    reasons: tuple[str, ...]
    chip: str | None
    priced_weeks: tuple[int, ...]
    fixtures: tuple[dict[str, int], ...]
    api_selling_absent: bool
    odds_trial: str = "not_sent"
    odds_remaining: str = ""
    odds_last_cost: str = ""
    live_rows: tuple[dict[str, Any], ...] = ()
    scorer_ran: bool = False
    copy_note: str = ""
    bench_gw: int | None = None
    schedule: tuple[tuple[str, int | None], ...] = ()
    outlooks: tuple[tuple[int, float, float, float, float], ...] = ()
    choice_field: str = ""
    dry_run: bool = False
    lineup_note: str = ""


def player_key(element: int) -> str:
    """Element id as the live season key."""
    return f"{SEASON}:{int(element)}"


def final_players(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The fifteen at the last played gameweek."""
    weeks = list(entry.get("gameweeks") or [])
    if not weeks:
        raise DeadlineError("entry has no gameweeks")
    last = max(weeks, key=lambda row: int(row["gw"]))
    players = list(last.get("xi") or []) + list(last.get("bench") or [])
    ids = [int(player["id"]) for player in players]
    if len(ids) != len(set(ids)):
        raise DeadlineError(f"GW{int(last['gw'])} lists a player twice")
    return players


def reconstruct_purchases(
    entry: Mapping[str, Any], gw1_value: Mapping[int, int]
) -> dict[int, int]:
    """Purchase price for the final owned set.

    A Gameweek 1 player keeps that week's price. A later buy keeps
    ``in_cost``. A sale of someone not owned, a Gameweek 1 player with
    no price, or a final player the transfers do not explain, raises.
    """
    owned: dict[int, int] = {}
    for player in entry.get("opening_squad") or []:
        pid = int(player["id"])
        if int(pid) not in gw1_value:
            name = player.get("name") or pid
            raise DeadlineError(f"{name} ({pid}) has no Gameweek 1 price")
        owned[pid] = int(gw1_value[int(pid)])
    transfers = sorted(entry.get("transfers") or [], key=lambda row: int(row["gw"]))
    for row in transfers:
        gw = int(row["gw"])
        out_id = int(row["out_id"])
        in_id = int(row["in_id"])
        if out_id not in owned:
            raise DeadlineError(f"GW{gw} sells {out_id} who is not owned")
        if in_id in owned:
            raise DeadlineError(f"GW{gw} buys {in_id} who is already owned")
        if "in_cost" not in row or row["in_cost"] is None:
            raise DeadlineError(f"GW{gw} buy {in_id} has no in_cost")
        del owned[out_id]
        owned[in_id] = int(row["in_cost"])
    final_ids = {int(player["id"]) for player in final_players(entry)}
    extra = set(owned) - final_ids
    missing = final_ids - set(owned)
    if extra or missing:
        raise DeadlineError(
            f"unexplained squad: extra {sorted(extra)} missing {sorted(missing)}"
        )
    return owned


def purchase_source(entry: Mapping[str, Any], element: int) -> str:
    opening = {int(player["id"]) for player in entry.get("opening_squad") or []}
    return "gw1" if int(element) in opening else "transfer"


def api_selling(entry: Mapping[str, Any]) -> dict[int, int] | None:
    """Selling prices on the final picks. None when the payload has none."""
    found: dict[int, int] = {}
    for player in final_players(entry):
        if player.get("selling_price") is None:
            continue
        found[int(player["id"])] = int(player["selling_price"])
    return found or None


def resolve_holdings(
    players: Sequence[Mapping[str, Any]],
    purchases: Mapping[int, int],
    current: Mapping[int, int],
    api_prices: Mapping[int, int] | None,
    sources: Mapping[int, str],
) -> list[Holding]:
    """Formula sell, replaced by the site price when the picks carry one."""
    rows: list[Holding] = []
    for player in players:
        pid = int(player["id"])
        if pid not in purchases:
            raise DeadlineError(f"{player.get('name') or pid} has no purchase price")
        if pid not in current:
            raise DeadlineError(f"{player.get('name') or pid} has no current price")
        purchase = int(purchases[pid])
        now = int(current[pid])
        formula = sell_price(purchase, now)
        if api_prices is not None and pid in api_prices:
            selling = int(api_prices[pid])
            origin = "api"
        else:
            selling = formula
            origin = "formula"
        rows.append(
            Holding(
                element=pid,
                key=player_key(pid),
                name=str(player.get("name") or pid),
                position=str(player.get("position") or ""),
                team=str(player.get("team") or ""),
                purchase=purchase,
                purchase_source=str(sources.get(pid) or "transfer"),
                current=now,
                formula_sell=formula,
                selling=selling,
                selling_source=origin,
            )
        )
    return rows


def submitted_line(
    holdings: Sequence[Holding],
    choice: Mapping[str, float],
    bank: int,
) -> str:
    """The owned fifteen. The XI and the captain are the captured ``ep_next``."""
    frame = pd.DataFrame(
        [
            {
                "player_id": row.key,
                "position": row.position,
                "score_xp": float(choice[row.key]),
            }
            for row in holdings
        ]
    )
    picked = live_xi(frame, "score_xp")
    names = {row.key: row.name for row in holdings}
    form = picked["formation"]
    shape = f"1-{form[0]}-{form[1]}-{form[2]}"
    captain = str(picked["captain"])
    vice = str(picked["vice"])

    def label(pid: str) -> str:
        if pid == captain:
            return f"{names.get(pid, pid)} (C)"
        if pid == vice:
            return f"{names.get(pid, pid)} (V)"
        return names.get(pid, pid)

    xi_ids = [str(pid) for pid in picked["xi"]["player_id"]]
    bench_ids = [str(pid) for pid in picked["bench"]["player_id"]]
    positions = [row.position for row in holdings]
    clubs = [row.team for row in holdings]
    xi_positions = [str(pos) for pos in picked["xi"]["position"]]
    structure = squad_legal(positions, clubs) and xi_legal(xi_positions) and int(bank) >= 0
    state = "legal" if structure else "not legal"
    xi_text = ", ".join(label(pid) for pid in xi_ids)
    bench_text = ", ".join(names.get(pid, pid) for pid in bench_ids)
    return (
        f"Submitted squad is the current 15, {state}, bank {int(bank)} tenths. "
        f"Formation {shape}. XI: {xi_text}. Bench: {bench_text}. "
        "No transfer and no hit are priced in this note. The chip is the logged judgement."
    )


def holdings_state(holdings: Sequence[Holding], bank: int, ft: int) -> SquadState:
    """Squad state for one rebuild. The site price wins when any row has one."""
    purchase = {row.key: row.purchase for row in holdings}
    selling = None
    if any(row.selling_source == "api" for row in holdings):
        selling = {row.key: row.selling for row in holdings}
    return SquadState(purchase=purchase, bank=int(bank), ft=int(ft), selling=selling)


def apply_live_minutes(
    ids: Sequence[int],
    supplied: Mapping[int, float] | None,
    last_minutes: Mapping[int, float],
) -> list[dict[str, Any]]:
    """Minutes for this live pricer.

    A value in ``supplied`` is used as written, including zero. An id the
    file does not list keeps ``last_minutes`` and is ``no_news``. An id
    with no appearance is 0 and ``no_history``. ``supplied`` of None means
    there was no file: the same fallback applies, and the caller records
    ``missing_minutes``.
    """
    rows: list[dict[str, Any]] = []
    for pid in ids:
        element = int(pid)
        if supplied is not None and element in supplied:
            rows.append(
                {"player_id": element, "xmi": float(supplied[element]), "source": "file"}
            )
        elif element in last_minutes:
            rows.append(
                {
                    "player_id": element,
                    "xmi": float(last_minutes[element]),
                    "source": "no_news",
                }
            )
        else:
            rows.append({"player_id": element, "xmi": 0.0, "source": "no_history"})
    return rows


def _week_fixtures(fixtures: Sequence[Mapping[str, Any]], gw: int) -> list[Mapping[str, Any]]:
    rows = []
    for fixture in fixtures:
        event = fixture.get("event")
        if event is None:
            continue
        if int(event) == int(gw):
            rows.append(fixture)
    return rows


def clubs_missing_line(
    odds: pd.DataFrame,
    fixtures: Sequence[Mapping[str, Any]],
    team_names: Mapping[int, str],
    gw: int,
) -> list[str]:
    """Clubs with a fixture and no opening 1X2. A week with no fixtures is empty."""
    week = _week_fixtures(fixtures, gw)
    if not week:
        return []
    pots = opening_pots_by_team_gw(odds, week, dict(team_names))
    missing: list[str] = []
    seen: set[str] = set()
    for fixture in week:
        for side in ("team_h", "team_a"):
            club_id = int(fixture[side])
            raw = team_names.get(club_id)
            label = raw or str(club_id)
            key = (int(gw), norm_team(raw) if raw else "")
            if raw and key in pots:
                continue
            if label not in seen:
                seen.add(label)
                missing.append(label)
    return missing


def line_status(
    odds: pd.DataFrame,
    fixtures: Sequence[Mapping[str, Any]],
    team_names: Mapping[int, str],
    gw: int,
) -> str:
    """``blank`` has no fixtures. ``priced`` joins every club. Otherwise the line is missing."""
    if not _week_fixtures(fixtures, gw):
        return "blank"
    if clubs_missing_line(odds, fixtures, team_names, gw):
        return "missing_opening_line"
    return "priced"


def readiness(status: str, minutes_file: bool) -> tuple[bool, tuple[str, ...]]:
    """A missing line or a missing minutes file means no chip is chosen."""
    reasons: list[str] = []
    if status == "missing_opening_line":
        reasons.append("missing_opening_line")
    elif status == "blank":
        reasons.append("blank_week")
    elif status != "priced":
        reasons.append(status)
    if not minutes_file:
        reasons.append("missing_minutes")
    return (not reasons), tuple(reasons)


def bench_for_transfers(plan: HalfPlan, gw: int) -> int | None:
    """Bench Boost week for the transfer search. Wildcard and Free Hit pass none."""
    if plan.chip in FREE_TRANSFER_CHIPS:
        return None
    return bench_week(plan, gw)


def price_and_plan(
    current_gw: int,
    weeks: Sequence[WeekInputs],
    played: Mapping[int, str] | None = None,
) -> HalfPlan:
    """Plan a half the caller has already priced. This does not invent a team rate."""
    return plan_half(int(current_gw), weeks, played=played)


def gameweek_values(logs: pd.DataFrame, gw: int) -> dict[int, int]:
    """Price in tenths at one gameweek. Two different prices for one player raise."""
    frame = logs.loc[pd.to_numeric(logs["gw"], errors="coerce") == int(gw)]
    out: dict[int, int] = {}
    for row in frame.itertuples(index=False):
        pid = int(row.player_id)
        value = int(row.value)
        if pid in out and out[pid] != value:
            raise DeadlineError(f"{pid} has two Gameweek {gw} prices")
        out[pid] = value
    return out


def last_observed_minutes(logs: pd.DataFrame) -> dict[int, float]:
    """Minutes in each player's latest stored gameweek. A double in that week is summed."""
    frame = logs.copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["gw"] = pd.to_numeric(frame["gw"], errors="coerce")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["player_id", "gw"])
    if frame.empty:
        return {}
    latest = frame.groupby("player_id")["gw"].transform("max")
    tail = frame.loc[frame["gw"] == latest]
    totals = tail.groupby("player_id")["minutes"].sum()
    return {int(pid): float(mins) for pid, mins in totals.items()}


def current_costs(bootstrap: Mapping[str, Any]) -> dict[int, int]:
    return {int(row["id"]): int(row["now_cost"]) for row in bootstrap["elements"]}


def team_names(bootstrap: Mapping[str, Any]) -> dict[int, str]:
    return {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}


def deadline_text(bootstrap: Mapping[str, Any], gw: int) -> str:
    for event in bootstrap.get("events") or []:
        if int(event.get("id") or 0) == int(gw):
            return str(event.get("deadline_time") or "")
    return ""


def fixture_calendar(
    fixtures: Sequence[Mapping[str, Any]], start: int, end: int
) -> list[dict[str, int]]:
    """Match counts. A double is a club with two or more fixtures that week."""
    rows = []
    for gw in range(int(start), int(end) + 1):
        week = _week_fixtures(fixtures, gw)
        counts: dict[int, int] = {}
        for fixture in week:
            for side in ("team_h", "team_a"):
                club = int(fixture[side])
                counts[club] = counts.get(club, 0) + 1
        rows.append(
            {
                "gw": int(gw),
                "matches": len(week),
                "clubs": len(counts),
                "doubles": sum(1 for n in counts.values() if n >= 2),
            }
        )
    return rows


def file_hash(path: Path) -> str:
    """First 16 hex characters of the file's SHA-256."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def _element_id(value: object) -> int:
    text = str(value)
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return int(text)


def load_minutes(path: Path, gw: int) -> dict[int, float]:
    """One gameweek from a ``player_id, gw, xmi`` file. Validation stays in ``load_xmi``."""
    from src.live.xmi import load_xmi

    frame = load_xmi(path, gw)
    out: dict[int, float] = {}
    for row in frame.itertuples(index=False):
        pid = _element_id(row.player_id)
        if pid in out:
            raise DeadlineError(f"duplicate minutes for {pid}")
        out[pid] = float(row.xmi)
    return out


def write_log(text: str, path: Path) -> None:
    if path.name in FORBIDDEN_NAMES:
        raise DeadlineError(f"refusing to write {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _money(tenths: int) -> str:
    return f"£{int(tenths) / 10:.1f}m"


def _price(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if number != number:
        return ""
    return f"{number:.2f}"


def _books(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if number != number:
        return ""
    return str(int(number))


def load_odds_frame(odds_path: Path, live_path: Path | None) -> pd.DataFrame:
    """Football-data history, then the live file so a live row wins on a clash."""
    frame = pd.read_csv(odds_path)
    if live_path is None or not Path(live_path).is_file():
        return frame
    live = pd.read_csv(live_path)
    if live.empty:
        return frame
    return pd.concat([frame, live], ignore_index=True)


def read_live_rows(path: Path | None) -> tuple[dict[str, Any], ...]:
    if path is None or not Path(path).is_file():
        return ()
    frame = pd.read_csv(path)
    rows = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                "gw": int(float(record["gw"])),
                "home": record.get("HomeTeam"),
                "away": record.get("AwayTeam"),
                "avg_h": record.get("AvgH"),
                "avg_d": record.get("AvgD"),
                "avg_a": record.get("AvgA"),
                "over": record.get("Avg>2.5"),
                "under": record.get("Avg<2.5"),
                "source": record.get("source"),
                "books": record.get("books"),
            }
        )
    return tuple(rows)


def read_trial(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {"reason": "not_sent", "remaining": "", "last": ""}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render(log: DeadlineLog) -> str:
    """The pre-deadline note. A stop names the missing line and chooses no chip."""
    played = ", ".join(f"GW{gw} {chip}" for gw, chip in log.chips_played) or "none"
    left = ", ".join(log.chips_left) or "none"
    lines = [
        f"# Gameweek {log.gw} deadline",
        "",
    ]
    if log.dry_run:
        lines.append(
            "This is a dry run. It is not the decision pair. "
            "The decision capture is the T-1h same-stamp file, and an earlier file is not a fallback."
        )
        lines.append("")
    chip_sentence = (
        "No chip was chosen." if log.chip is None else f"The chip this week is {log.chip}."
    )
    lines.append(
        f"{log.team_name} (entry {log.entry_id}) at the Gameweek {log.gw} "
        f"deadline, {log.deadline or 'time not on the snapshot'}. "
        f"This note was written before that deadline. {chip_sentence}"
    )
    lines.append("")
    if "missing_opening_line" in log.reasons:
        lines.append(
            "This gameweek still has a club with no 1X2. The run stops there. "
            "It does not invent a team rate from the Asian handicap or from the "
            "season table, and it does not call the half-season plan."
        )
        lines.append("")
    elif log.line_status == "priced" and not log.scorer_ran:
        lines.append(
            "Every club in this gameweek has a 1X2 on the live file. "
            "The scorer is ready and was not run because the minutes file is absent."
        )
        lines.append("")
    elif log.scorer_ran and log.choice_field == "ep_next":
        lines.append(
            "The squad this week was chosen by the captured ep_next. "
            "score_xp was logged beside it and did not choose the squad. "
            "Later horizon weeks have no ep_next capture, so they add nothing "
            "to the chip sum and are not filled from score_xp. "
            "Minutes came from the file, a zero stayed a zero, and a player "
            "the file omits kept his last observed minutes. "
            "The transfer search was not run. "
            "The chip is a logged judgement and does not count as a chip rule. "
            "A chip named in this note is the plan, not the submission. "
            "An injury flag is the minutes file, and a written zero stays zero. "
            "No manual override is recorded."
        )
        lines.append("")
    elif log.scorer_ran:
        lines.append(
            "The scorer called the same one-match formula as score_xp. "
            "Minutes came from the file, a zero stayed a zero, and a player "
            "the file omits kept his last observed minutes. Shot shares stayed "
            "on the deadline. Each priced week uses that week's opening pot, "
            "and only the first pot when a club has two fixtures. "
            "One rebuild was paid from the bank plus sales. "
            "The transfer search was not run."
        )
        if log.copy_note:
            lines.append(log.copy_note)
        if log.chip is not None and log.copy_note:
            lines.append(
                "The wildcard hurdle adds only the weeks with their own opening "
                "line. A week that repeats the last priced step adds nothing to "
                "that sum and nothing to the schedule. The margin stays 16."
            )
        lines.append("")
    if log.odds_trial in {"no_betfair_app_key", "no_key"}:
        lines.append(
            "Betfair was not called. No app key is set. "
            "The live 1X2 stays empty until Exchange credentials are present."
        )
        lines.append("")
    elif log.odds_trial == "betfair":
        lines.append(
            "Live 1X2 and over/under 2.5 come from the Betfair Exchange only: "
            "unweighted back/lay mids, simplex-normalised to fair decimals, "
            "with tiered liquidity shrinkage. Odds API and ESPN are not used."
        )
        lines.append("")
    elif log.odds_trial in {"betfair_error", "betfair_geo_blocked", "error"}:
        lines.append(
            "The Betfair request failed. Odds API fallback is forbidden, so no "
            "bookmaker line was written. "
            f"Reason: {log.odds_trial}."
        )
        lines.append("")
    if "missing_minutes" in log.reasons:
        lines.append(
            "No minutes file was passed, so nothing was hashed. A player with a "
            "stored appearance keeps his last observed minutes and is labelled "
            "no_news. That is the live pricer's fallback when a file omits him. "
            "It is not a news sheet, and it is not a reason to pick a chip."
        )
        lines.append("")
    if log.api_selling_absent:
        lines.append(
            "The stored picks have no selling price. Every row uses the "
            "rules-module formula: half the rise, rounded down, and the full fall."
        )
    else:
        mismatched = [row for row in log.holdings if row.mismatch]
        if mismatched:
            names = ", ".join(row.name for row in mismatched)
            lines.append(
                "Where the site's selling price differs from the formula, the "
                f"site price is the one a chip would pay. Differing rows: {names}."
            )
        else:
            lines.append(
                "The site's selling price is on the picks and matches the formula."
            )
    lines.append("")
    lines.append(
        f"Bank {log.bank} tenths ({_money(log.bank)}), {log.ft} free transfer. "
        f"Chips already played: {played}. Still available: {left}. "
        "The model's Gameweek 1–5 climb is a different squad and is not the squad in this note."
    )
    lines.append("")
    lines.append("## Holdings")
    lines.append("")
    lines.append("Prices are tenths of £1m.")
    lines.append("")
    lines.append("| Player | Pos | Club | Purchase | Source | Current | Formula sell | Used |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in log.holdings:
        used = f"{row.selling} ({row.selling_source})"
        lines.append(
            f"| {row.name} | {row.position} | {row.team} | {row.purchase} | "
            f"{row.purchase_source} | {row.current} | {row.formula_sell} | {used} |"
        )
    lines.append("")
    lines.append("## Minutes")
    lines.append("")
    if log.minutes_hash:
        lines.append(
            f"File `{log.minutes_file}`, SHA-256 prefix `{log.minutes_hash}`."
        )
    else:
        lines.append("Minutes file: absent.")
    lines.append("")
    lines.append("| Player | xmi | Source |")
    lines.append("| --- | --- | --- |")
    by_id = {row.element: row.name for row in log.holdings}
    for row in log.minutes:
        lines.append(
            f"| {by_id.get(int(row['player_id']), row['player_id'])} | "
            f"{float(row['xmi']):.0f} | {row['source']} |"
        )
    lines.append("")
    lines.append("## Line")
    lines.append("")
    lines.append(f"Status: {log.line_status}.")
    if log.missing_clubs:
        lines.append("")
        lines.append("Clubs with a fixture and no opening 1X2: " + ", ".join(log.missing_clubs) + ".")
    if log.live_rows:
        lines.append("")
        lines.append("| GW | Match | Source | Books | 1X2 | Over 2.5 | Under 2.5 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        ordered = sorted(log.live_rows, key=lambda row: (int(row["gw"]), str(row["home"])))
        for row in ordered:
            one = f"{_price(row['avg_h'])} / {_price(row['avg_d'])} / {_price(row['avg_a'])}"
            books = _books(row.get("books"))
            lines.append(
                f"| {int(row['gw'])} | {row['home']} vs {row['away']} | {row['source']} | "
                f"{books} | {one} | {_price(row['over']) or 'blank'} | {_price(row['under']) or 'blank'} |"
            )
    lines.append("")
    lines.append("## Plan")
    lines.append("")
    lines.append(
        f"Chip: {'not chosen' if log.chip is None else log.chip}. "
        f"Stops: {', '.join(log.reasons) or 'none'}."
    )
    if log.lineup_note:
        lines.append(log.lineup_note)
    if log.priced_weeks:
        weeks = ", ".join(f"GW{gw}" for gw in log.priced_weeks)
        lines.append(f"Priced weeks: {weeks}.")
    else:
        lines.append("Priced weeks: none.")
    if log.scorer_ran:
        bench = "none" if log.bench_gw is None else f"GW{log.bench_gw}"
        lines.append(
            f"Free Hit hurdle {FH_MARGIN:g} and Wildcard hurdle {WC_MARGIN:g} "
            f"were applied. The bench week recorded for a later transfer search "
            f"is {bench}. No transfer search was run."
        )
        if log.schedule:
            parts = [
                f"{name} {'none' if gw is None else f'GW{gw}'}"
                for name, gw in log.schedule
            ]
            lines.append("Schedule: " + ", ".join(parts) + ".")
            if log.copy_note:
                lines.append(
                    "A copied week adds nothing, so a chip that would sit only "
                    "on a copied week is left unset. Free Hit is checked against "
                    "12 only when it is the chip this week."
                )
        if log.outlooks:
            lines.append("")
            lines.append("| GW | Held XI | Held bench | Rebuilt XI | Free hit |")
            lines.append("| --- | --- | --- | --- | --- |")
            for gw, held, bench_xp, rebuilt, fh in log.outlooks:
                lines.append(
                    f"| {gw} | {held:.2f} | {bench_xp:.2f} | {rebuilt:.2f} | {fh:.2f} |"
                )
    else:
        lines.append(
            f"Free Hit hurdle {FH_MARGIN:g} and Wildcard hurdle {WC_MARGIN:g} "
            "were not applied. No transfer search was run."
        )
    lines.append("")
    lines.append(f"## Fixtures through Gameweek {half_end(log.gw)}")
    lines.append("")
    lines.append("| GW | Matches | Clubs | Doubles |")
    lines.append("| --- | --- | --- | --- |")
    for row in log.fixtures:
        lines.append(
            f"| {row['gw']} | {row['matches']} | {row['clubs']} | {row['doubles']} |"
        )
    lines.append("")
    if log.scorer_ran:
        lines.append(
            "Doubles stay one fixture. The first pot is the one that was scored. "
            "A week with no clubs is zero and is not the week later scores copy."
        )
    else:
        lines.append(
            "Doubles stay one fixture. The live file stores the 1X2 it could join. "
            "Those rows were not passed to the half-season plan."
        )
    lines.append("")
    return "\n".join(lines)


def collect(
    *,
    entry_path: Path = ENTRY_PATH,
    log_path: Path = LOG_PATH,
    odds_path: Path = ODDS_PATH,
    bootstrap_path: Path = BOOTSTRAP_PATH,
    fixtures_path: Path = FIXTURES_PATH,
    minutes_path: Path | None = None,
    live_path: Path | None = None,
    trial_path: Path | None = TRIAL_META,
    gw: int = DECISION_GW,
    dry_run: bool = False,
    decision_file: Path | None = None,
) -> DeadlineLog:
    """Read the stored files. A Betfair 1X2 is used in front of the historical file.

    The decision score is the T−1h ``ep_next`` slot. A dry run names its own
    file and does not become that slot. The frozen holdout slate is not read.
    """
    if dry_run and decision_file is None:
        raise DeadlineError("a dry run names its capture")
    if decision_file is not None and not dry_run:
        raise DeadlineError("the decision capture is the T-1h slot")
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    logs = pd.read_csv(log_path)
    from src.live.betfair_props import resolve_live_book

    book = resolve_live_book(int(gw), live_path)
    if book is not None and (trial_path is None or trial_path == TRIAL_META):
        meta_candidate = book.parent / "betfair_meta.json"
        if meta_candidate.is_file():
            trial_path = meta_candidate
    live_path = book
    odds = load_odds_frame(odds_path, live_path)
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    players = final_players(entry)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    sources = {pid: purchase_source(entry, pid) for pid in purchases}
    site = api_selling(entry)
    holdings = resolve_holdings(
        players, purchases, current_costs(bootstrap), site, sources
    )
    state = holdings_state(holdings, int(entry["bank"]), int(entry["ft_for_next"]))
    names = team_names(bootstrap)
    status = line_status(odds, fixtures, names, gw)
    missing = tuple(clubs_missing_line(odds, fixtures, names, gw))
    minutes_file = None
    minutes_hash = None
    supplied: dict[int, float] | None = None
    if minutes_path is not None and Path(minutes_path).is_file():
        minutes_file = str(minutes_path)
        minutes_hash = file_hash(Path(minutes_path))
        supplied = load_minutes(Path(minutes_path), gw)
    ready, reasons = readiness(status, minutes_file is not None)
    minutes = tuple(
        apply_live_minutes(
            [row.element for row in holdings],
            supplied,
            last_observed_minutes(logs),
        )
    )
    played = tuple(
        (int(row["gw"]), str(row["chip"])) for row in entry.get("chips_played") or []
    )
    trial = read_trial(trial_path)
    chip = None
    priced: tuple[int, ...] = ()
    scorer_ran = False
    note = ""
    bench_gw = None
    schedule: tuple[tuple[str, int | None], ...] = ()
    outlooks: tuple[tuple[int, float, float, float, float], ...] = ()
    lineup_note = ""
    if ready:
        from src.live.scorer import player_key as score_key
        from src.live.scorer import price_half

        resolved = apply_live_minutes(
            [int(element["id"]) for element in bootstrap["elements"]],
            supplied,
            last_observed_minutes(logs),
        )
        minute_map = {
            score_key(int(row["player_id"])): float(row["xmi"]) for row in resolved
        }
        from src.live.scorer import load_ep_next

        if dry_run:
            choice = load_ep_next(int(gw), path=Path(decision_file))  # type: ignore[arg-type]
        else:
            choice = load_ep_next(int(gw))
        from src.live.betfair_props import discover_betfair_artifacts

        artifacts = discover_betfair_artifacts(int(gw))
        scored = price_half(
            gw=int(gw),
            logs=logs,
            odds=odds,
            fixtures=fixtures,
            bootstrap=bootstrap,
            state=state,
            minutes=minute_map,
            played={week: chip_name for week, chip_name in played},
            choice=choice,
            artifacts_dir=artifacts,
        )
        chip = scored.plan.chip
        priced = scored.line_weeks
        scorer_ran = True
        note = scored.copy_note
        bench_gw = bench_for_transfers(scored.plan, int(gw))
        schedule = tuple(
            (str(name), None if week is None else int(week))
            for name, week in scored.plan.schedule.items()
        )
        outlooks = tuple(
            (
                int(row.gw),
                float(row.held.xi_xp),
                float(row.held.bench_xp),
                float(row.rebuilt.xi_xp if row.rebuilt is not None else 0.0),
                float(row.fh_xi or 0.0),
            )
            for row in scored.weeks
        )
        lineup_note = submitted_line(holdings, choice, int(entry["bank"]))
    return DeadlineLog(
        entry_id=int(entry["entry_id"]),
        team_name=str(entry.get("team_name") or ""),
        gw=int(gw),
        deadline=deadline_text(bootstrap, gw),
        bank=int(entry["bank"]),
        ft=int(entry["ft_for_next"]),
        chips_played=played,
        chips_left=tuple(str(chip) for chip in entry.get("chips_left") or []),
        holdings=tuple(holdings),
        state=state,
        minutes_file=minutes_file,
        minutes_hash=minutes_hash,
        minutes=minutes,
        line_status=status,
        missing_clubs=missing,
        reasons=reasons,
        chip=chip,
        priced_weeks=priced,
        fixtures=tuple(fixture_calendar(fixtures, gw, half_end(gw))),
        api_selling_absent=site is None,
        odds_trial=str(trial.get("reason") or "not_sent"),
        odds_remaining=str(trial.get("remaining") or ""),
        odds_last_cost=str(trial.get("last") or ""),
        live_rows=read_live_rows(live_path),
        scorer_ran=scorer_ran,
        copy_note=note,
        bench_gw=bench_gw,
        schedule=schedule,
        outlooks=outlooks,
        choice_field="ep_next" if scorer_ran else "",
        dry_run=dry_run,
        lineup_note=lineup_note,
    )


def run(
    report_path: Path = REPORT_PATH,
    **kwargs: Any,
) -> DeadlineLog:
    """Write the deadline note. The published reports are not this path."""
    log = collect(**kwargs)
    write_log(render(log), Path(report_path))
    return log


def main() -> None:
    """Read the deadline note. Does not refresh lines or touch the holdout slate."""
    log = run()
    print(
        f"GW{log.gw} {log.team_name}: {', '.join(log.reasons) or 'ready'}; "
        f"chip {'not chosen' if log.chip is None else log.chip}"
    )


if __name__ == "__main__":
    main()
