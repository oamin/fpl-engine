"""One live numagent run for the next deadline.

The order is fixed. Minutes are compiled from ``news_packets`` into
``gwNN/xmi_t1.csv``. The Exchange book is ``gwNN/betfair_t1`` and nowhere
else. ``price_half`` then runs with ``choice=None``, so the squad is chosen
on ``score_xp`` and later weeks stay ``forecast_xp``. ``--lock`` appends one
row to the numeric ledger only when the clock is before the deadline and
that gameweek has no row yet.

The dossier under ``string_sources`` is not a minutes input. A missing
Exchange file stops the run. An older ``betfair_*`` folder and
``data/live/gw_lines.csv`` are not opened.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from src.live.freeze_week import (
    ProvenanceViolationError,
    build_freeze_row,
    sha256_file,
    write_deadline_freeze,
)
from src.live.news_packets import (
    compile_player_xmi,
    load_gameweek_packets,
    packets_for_player,
)
from src.live.news_tags import prior_minutes, published_before
from src.live.plan import live_xi
from src.live.scorer import SCORE_COL
from src.live.t1_inputs import (
    FROZEN_LINES,
    FROZEN_MINUTES,
    exchange_dir,
    minutes_sheet,
    refuse_frozen,
)

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data" / "predictions" / "2026-27"
HOLDOUT_WEEKS = ROOT / "data" / "holdout" / "2026-27" / "player_gw"
EXCHANGE_FILES = ("gw_lines.csv", "betfair_to_score.json", "outrights_ranks.json")
LIVE_DIR = (ROOT / "data" / "live").resolve()


class SessionError(RuntimeError):
    """The live run cannot continue, and it has not locked a row."""


PullFn = Callable[[Path], int]


@dataclass(frozen=True)
class Lineup:
    squad_15: tuple[str, ...]
    starting_11: tuple[str, ...]
    captain: str
    vice_captain: str
    bench_order: tuple[str, ...]


@dataclass(frozen=True)
class NumagentPick:
    """Squad chosen on ``score_xp``, plus the two counterfactual elevens."""

    chip: str | None
    decision: Lineup
    hold_starting_11: tuple[str, ...]
    hold_captain: str
    ft1_starting_11: tuple[str, ...]
    ft1_captain: str
    ft1_move: str
    decision_xi_c_score_xp: float
    hold_xi_c_score_xp: float
    ft1_xi_c_score_xp: float
    step_scores: dict[int, dict[str, float]]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gw_folder(gw: int, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else ROOT
    return base / "data" / "predictions" / "2026-27" / f"gw{int(gw):02d}"


def manifest_path(gw: int, root: Path | None = None) -> Path:
    return gw_folder(gw, root) / "session_numagent.json"


def _under_live(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == LIVE_DIR or LIVE_DIR in resolved.parents


def assert_exchange_folder(folder: Path) -> None:
    """The only legal book is ``gwNN/betfair_t1``, outside ``data/live``."""
    if folder.name != "betfair_t1":
        raise SessionError(f"exchange folder must be betfair_t1, got {folder.name}")
    if _under_live(folder):
        raise SessionError(f"refusing data/live exchange folder {folder}")
    sheet = folder / "gw_lines.csv"
    if sheet.resolve() == FROZEN_LINES:
        raise SessionError(f"refusing frozen holdout file {sheet}")


def book_is_ready(folder: Path) -> bool:
    return all((folder / name).is_file() for name in EXCHANGE_FILES)


def _clear_partial_book(folder: Path) -> None:
    """Drop an incomplete trio so a later score cannot read a half book."""
    if book_is_ready(folder):
        return
    for name in EXCHANGE_FILES:
        path = folder / name
        if path.is_file():
            path.unlink()


def resolve_exchange(gw: int, *, root: Path | None = None, pull: PullFn | None = None) -> Path:
    """Reuse ``betfair_t1`` when it is complete. Otherwise pull into that folder.

    A non-zero pull removes any partial trio. Older ``betfair_*`` folders are
    not candidates.
    """
    folder = exchange_dir(gw) if root is None else gw_folder(gw, root) / "betfair_t1"
    assert_exchange_folder(folder)
    refuse_frozen(folder / "gw_lines.csv")
    if book_is_ready(folder):
        return folder
    if pull is None:
        pull = pull_betfair
    code = int(pull(folder))
    if code != 0 or not book_is_ready(folder):
        _clear_partial_book(folder)
        raise SessionError(f"Exchange pull for GW{int(gw)} failed ({code})")
    return folder


def pull_betfair(folder: Path) -> int:
    """Call ``betfair_lines`` into ``folder``. A 403 comes back as a non-zero code."""
    from src.live.betfair_lines import main as betfair_main

    assert_exchange_folder(folder)
    return int(betfair_main(["--out", str(folder)]))


def _ledger_has_gw(path: Path, gw: int) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and int(row.get("gw") or 0) == int(gw):
            return True
    return False


def _history(gw: int, history_dir: Path) -> dict[int, list[float]]:
    """One minutes total per finished week, in week order, zeros included."""
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for week in range(1, int(gw)):
        path = history_dir / f"gw{int(week):02d}.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        for column in ("player_id", "gw", "minutes"):
            if column not in frame.columns:
                raise SessionError(f"{path} is missing {column}")
        frames.append(frame.loc[:, ["player_id", "gw", "minutes"]])
    if missing:
        raise SessionError("finished minutes are missing: " + ", ".join(missing))
    if not frames:
        return {}
    whole = pd.concat(frames, ignore_index=True)
    whole["player_id"] = pd.to_numeric(whole["player_id"], errors="coerce")
    whole["gw"] = pd.to_numeric(whole["gw"], errors="coerce")
    whole["minutes"] = pd.to_numeric(whole["minutes"], errors="coerce").fillna(0.0)
    whole = whole.dropna(subset=["player_id", "gw"])
    grouped = whole.groupby(["player_id", "gw"], as_index=False)["minutes"].sum()
    series: dict[int, dict[int, float]] = {}
    for row in grouped.itertuples(index=False):
        series.setdefault(int(row.player_id), {})[int(row.gw)] = float(row.minutes)
    out: dict[int, list[float]] = {}
    for pid, by_week in series.items():
        out[pid] = [float(by_week.get(week, 0.0)) for week in range(1, int(gw))]
    return out


def compile_minutes(
    gw: int,
    *,
    deadline_utc: str,
    root: Path | None = None,
    bootstrap_path: Path,
    history_dir: Path | None = None,
    ledger: Path | None = None,
    sources_path: Path | None = None,
) -> Path:
    """Write ``xmi_t1.csv`` from packets and finished weeks.

    A second call raises and leaves the file in place. ``sources_path`` is
    refused: the string dossier is not an input. A gameweek already on the
    ledger is refused.
    """
    if sources_path is not None:
        raise SessionError("string_sources are not an xmi input")
    base = Path(root) if root is not None else ROOT
    sheet = minutes_sheet(gw) if root is None else gw_folder(gw, base) / "xmi_t1.csv"
    refuse_frozen(sheet)
    if sheet.resolve() == FROZEN_MINUTES:
        raise SessionError(f"refusing frozen holdout file {sheet}")
    if sheet.is_file():
        raise SessionError(f"xmi_t1 already exists: {sheet}")
    ledger_path = ledger if ledger is not None else (PREDICTIONS / "week_freeze.jsonl")
    if root is not None and ledger is None:
        ledger_path = base / "data" / "predictions" / "2026-27" / "week_freeze.jsonl"
    if _ledger_has_gw(ledger_path, gw):
        raise SessionError(f"GW{int(gw)} is locked; xmi_t1 is not recompiled")
    history_root = history_dir or (base / "data" / "holdout" / "2026-27" / "player_gw")
    if history_dir is None and root is None:
        history_root = HOLDOUT_WEEKS
    if _under_live(Path(bootstrap_path)):
        raise SessionError("bootstrap for the minutes sheet must sit in the gameweek folder")
    bootstrap = json.loads(Path(bootstrap_path).read_text(encoding="utf-8"))
    from src.live.fpl_snapshot import ELEMENT

    packets = load_gameweek_packets(
        int(gw),
        deadline_utc=deadline_utc,
        bootstrap=bootstrap,
        root=base,
        include_synthetic_fpl=True,
    )
    history = _history(int(gw), history_root)
    rows: list[dict[str, Any]] = []
    for element in bootstrap.get("elements") or []:
        pid = int(element["id"])
        position = ELEMENT[int(element["element_type"])]
        series = history.get(pid, [])
        chance_raw = element.get("chance_of_playing_next_round")
        try:
            chance = None if chance_raw is None else float(chance_raw)
        except (TypeError, ValueError):
            chance = None
        compiled = compile_player_xmi(
            player_id=pid,
            name=str(element.get("web_name") or pid),
            position=position,
            prior=prior_minutes(series),
            status=None if element.get("status") is None else str(element.get("status")),
            chance=chance,
            packets=packets_for_player(packets, pid),
            minutes=series,
        )
        xmi = compiled["xmi_compiled"]
        if xmi is None:
            xmi = 0.0
        rows.append(
            {
                "player_id": pid,
                "gw": int(gw),
                "xmi": float(xmi),
                "tag": str(compiled["tag"]),
            }
        )
    rows.sort(key=lambda row: int(row["player_id"]))
    sheet.parent.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["player_id", "gw", "xmi", "tag"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return sheet


def write_step_scores(
    gw: int,
    step_scores: Mapping[int, Mapping[str, float]],
    *,
    stamp: str,
    root: Path | None = None,
) -> Path:
    """Every priced week. The deadline week is ``score_xp``; later weeks are ``forecast_xp``."""
    folder = gw_folder(gw, root)
    path = folder / f"{stamp}.csv"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    folder.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for week in sorted(int(key) for key in step_scores):
        column = "score_xp" if int(week) == int(gw) else "forecast_xp"
        for pid, score in sorted(step_scores[int(week)].items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "player_id": str(pid),
                    "gw": int(week),
                    "score": float(score),
                    "column": column,
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["player_id", "gw", "score", "column"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def lineup_points(
    starting: Sequence[str],
    captain: str,
    scores: Mapping[str, float],
) -> float:
    """Eleven plus the captain once more. A missing price is zero."""
    total = sum(float(scores.get(str(pid), 0.0)) for pid in starting)
    return total + float(scores.get(str(captain), 0.0))


def official_capture(gw: int, root: Path | None = None) -> Path:
    folder = gw_folder(gw, root)
    found = sorted(folder.glob("official_*_t1.csv"))
    if not found:
        raise SessionError(f"official T-1 capture is missing in {folder}")
    return found[-1]


def load_ep_next(path: Path, gw: int) -> dict[str, float]:
    frame = pd.read_csv(path)
    for column in ("player_id", "gw", "official_xp", "source_field"):
        if column not in frame.columns:
            raise SessionError(f"{path} is missing {column}")
    use = frame.loc[pd.to_numeric(frame["gw"], errors="coerce") == int(gw)]
    use = use.loc[use["source_field"].astype(str) == "ep_next"]
    if use.empty:
        raise SessionError(f"{path} has no ep_next rows for GW{int(gw)}")
    scores: dict[str, float] = {}
    for row in use.itertuples(index=False):
        scores[str(row.player_id)] = float(row.official_xp)
    return scores


def _ids(values: Sequence[str]) -> list[str]:
    return [str(pid) for pid in values]


def lock_from_pick(
    pick: NumagentPick,
    *,
    gw: int,
    deadline_utc: str,
    frozen_at_utc: str,
    official_path: Path,
    minutes_path: Path,
    lines_path: Path,
    scores_path: Path,
    ledger: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Build the freeze row from the score_xp squads and the official ``ep_next`` file.

    The clock is checked first. A late clock or a second gameweek writes
    nothing.
    """
    if not published_before(frozen_at_utc, deadline_utc):
        raise SessionError("frozen_at is not before the deadline")
    if _ledger_has_gw(ledger, gw):
        raise ProvenanceViolationError(f"GW{int(gw)} already has a freeze row")
    ep_next = load_ep_next(official_path, gw)
    decision = pick.decision
    base = Path(root) if root is not None else ROOT

    def _rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            return str(path)

    decision_ep = lineup_points(decision.starting_11, decision.captain, ep_next)
    hold_ep = lineup_points(pick.hold_starting_11, pick.hold_captain, ep_next)
    ft1_ep = lineup_points(pick.ft1_starting_11, pick.ft1_captain, ep_next)
    row = build_freeze_row(
        gw=int(gw),
        deadline_utc=deadline_utc,
        frozen_at_utc=frozen_at_utc,
        provenance={
            "official_capture_file": _rel(official_path),
            "official_sha256": sha256_file(official_path),
            "minutes_sha256": sha256_file(minutes_path),
            "odds_source": "betfair",
            "odds_manifest_sha256": sha256_file(lines_path),
            "pool_forecast_sha256": sha256_file(scores_path),
        },
        decision={
            "chip_played": pick.chip,
            "squad_15": _ids(decision.squad_15),
            "starting_11": _ids(decision.starting_11),
            "captain": str(decision.captain),
            "vice_captain": str(decision.vice_captain),
            "bench_order": _ids(decision.bench_order),
        },
        counterfactuals={
            "hold_starting_11": _ids(pick.hold_starting_11),
            "hold_captain": str(pick.hold_captain),
            "ft1_starting_11": _ids(pick.ft1_starting_11),
            "ft1_captain": str(pick.ft1_captain),
            "ft1_move": str(pick.ft1_move),
        },
        forecast_expected_points={
            "decision_xi_c_ep_next": decision_ep,
            "decision_xi_c_score_xp": float(pick.decision_xi_c_score_xp),
            "hold_xi_c_ep_next": hold_ep,
            "hold_xi_c_score_xp": float(pick.hold_xi_c_score_xp),
            "ft1_xi_c_ep_next": ft1_ep,
            "ft1_xi_c_score_xp": float(pick.ft1_xi_c_score_xp),
        },
    )
    write_deadline_freeze(row, ledger)
    return row


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SessionError(f"{path} is not an object")
    return payload


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    existing = _read_manifest(path)
    if existing is not None and existing.get("locked") is True:
        raise SessionError(f"locked manifest is not overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lineup_from_state(state: Any, pool: pd.DataFrame, scores: Mapping[str, float]) -> tuple[Lineup, float]:
    from src.models.half_plan_scores import squad_outlook

    frame = pool.drop_duplicates("player_id", keep="first").copy()
    frame["player_id"] = frame["player_id"].astype(str)
    owned = frame.loc[frame["player_id"].isin(state.ids())].copy()
    owned[SCORE_COL] = [float(scores.get(str(pid), 0.0)) for pid in owned["player_id"]]
    view = live_xi(owned, SCORE_COL)
    starting = tuple(str(pid) for pid in view["xi"]["player_id"])
    bench = tuple(str(pid) for pid in view["bench"]["player_id"])
    squad = starting + bench
    outlook = squad_outlook(
        frame,
        set(state.ids()),
        {str(pid): float(value) for pid, value in scores.items()},
        SCORE_COL,
    )
    lineup = Lineup(
        squad_15=squad,
        starting_11=starting,
        captain=str(view["captain"]),
        vice_captain=str(view["vice"]),
        bench_order=bench,
    )
    return lineup, float(outlook.xi_xp)


def _move_text(before: set[str], after: set[str], names: Mapping[str, str]) -> str:
    sold = sorted(before - after)
    bought = sorted(after - before)
    if not sold and not bought:
        return "hold"
    parts: list[str] = []
    if sold:
        parts.append("sell " + ", ".join(f"{pid} {names.get(pid, pid)}" for pid in sold))
    if bought:
        parts.append("buy " + ", ".join(f"{pid} {names.get(pid, pid)}" for pid in bought))
    return "; ".join(parts)


def score_deadline(
    *,
    gw: int,
    logs: pd.DataFrame,
    odds: pd.DataFrame,
    fixtures: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    state: Any,
    minutes: Mapping[str, float],
    played: Mapping[int, str] | None,
    artifacts_dir: Path,
    choice: Mapping[str, float] | None = None,
) -> Any:
    """``choice`` stays empty. The artifacts folder is the T−1 book."""
    if choice is not None:
        raise SessionError("the live session chooses the squad on score_xp")
    assert_exchange_folder(artifacts_dir)
    if not book_is_ready(artifacts_dir):
        raise SessionError(f"exchange book is incomplete: {artifacts_dir}")
    from src.live.scorer import price_half

    return price_half(
        gw=int(gw),
        logs=logs,
        odds=odds,
        fixtures=fixtures,
        bootstrap=bootstrap,
        state=state,
        minutes=minutes,
        played=played,
        choice=None,
        artifacts_dir=artifacts_dir,
    )


def select_on_score_xp(
    gw: int,
    *,
    minutes_path: Path,
    exchange: Path,
    bootstrap_path: Path,
    fixtures_path: Path,
    entry_path: Path,
    logs_path: Path,
    odds_path: Path,
) -> NumagentPick:
    """Price the half on ``score_xp`` and keep the hold and one-transfer elevens."""
    from src.live.deadline import (
        apply_live_minutes,
        current_costs,
        final_players,
        gameweek_values,
        holdings_state,
        last_observed_minutes,
        load_minutes,
        load_odds_frame,
        purchase_source,
        reconstruct_purchases,
        resolve_holdings,
        team_names,
    )
    from src.live.scorer import (
        build_pool,
        clubs_from_fixtures,
        deadline_shares,
        player_key,
        roster_from_bootstrap,
    )
    from src.models.season_climb_ft import choose_transfers, rebuild_squad

    assert_exchange_folder(exchange)
    if _under_live(bootstrap_path) or _under_live(fixtures_path) or _under_live(minutes_path):
        raise SessionError("the live score does not read data/live")
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    logs = pd.read_csv(logs_path)
    odds = load_odds_frame(odds_path, exchange / "gw_lines.csv")
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    players = final_players(entry)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    sources = {pid: purchase_source(entry, pid) for pid in purchases}
    holdings = resolve_holdings(players, purchases, current_costs(bootstrap), None, sources)
    state = holdings_state(holdings, int(entry["bank"]), int(entry["ft_for_next"]))
    supplied = load_minutes(minutes_path, int(gw))
    resolved = apply_live_minutes(
        [int(element["id"]) for element in bootstrap["elements"]],
        supplied,
        last_observed_minutes(logs),
    )
    minute_map = {player_key(int(row["player_id"])): float(row["xmi"]) for row in resolved}
    played = {int(row["gw"]): str(row["chip"]) for row in entry.get("chips_played") or []}
    scored = score_deadline(
        gw=int(gw),
        logs=logs,
        odds=odds,
        fixtures=fixtures,
        bootstrap=bootstrap,
        state=state,
        minutes=minute_map,
        played=played,
        artifacts_dir=exchange,
        choice=None,
    )
    week_scores = {str(pid): float(value) for pid, value in scored.step_scores[int(gw)].items()}
    names = team_names(bootstrap)
    shares = deadline_shares(logs, int(gw))
    roster = roster_from_bootstrap(bootstrap)
    pool = build_pool(roster, shares, minute_map, set(state.ids()))
    frame = pool.drop_duplicates("player_id", keep="first").copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame[SCORE_COL] = [float(week_scores.get(str(pid), 0.0)) for pid in frame["player_id"]]
    line_weeks = [int(week) for week in scored.line_weeks]
    if int(gw) not in line_weeks:
        raise SessionError(f"GW{int(gw)} is not a priced week")
    clubs = clubs_from_fixtures(fixtures, names, int(gw), max(line_weeks))
    score_by_gw = {
        int(week): {str(pid): float(value) for pid, value in scored.step_scores[int(week)].items()}
        for week in line_weeks
    }
    roster_by_gw = {int(week): set(frame["player_id"].astype(str)) for week in line_weeks}
    club_map = {int(week): set(clubs[int(week)]) for week in line_weeks if int(week) in clubs}
    rebuilt = rebuild_squad(state, frame, SCORE_COL)
    ft_state, _n_tx, _hits = choose_transfers(
        state,
        frame,
        SCORE_COL,
        int(gw),
        line_weeks,
        roster_by_gw,
        score_by_gw=score_by_gw,
        clubs=club_map,
        bench_gw=None,
    )
    chip = scored.plan.chip
    decision_state = rebuilt if chip in {"wildcard", "free_hit"} else ft_state
    decision, decision_xp = _lineup_from_state(decision_state, frame, week_scores)
    hold, hold_xp = _lineup_from_state(state, frame, week_scores)
    ft1, ft1_xp = _lineup_from_state(ft_state, frame, week_scores)
    web = {
        player_key(int(element["id"])): str(element.get("web_name") or element["id"])
        for element in bootstrap["elements"]
    }
    steps = {
        int(week): {str(pid): float(value) for pid, value in scores.items()}
        for week, scores in scored.step_scores.items()
    }
    return NumagentPick(
        chip=chip,
        decision=decision,
        hold_starting_11=hold.starting_11,
        hold_captain=hold.captain,
        ft1_starting_11=ft1.starting_11,
        ft1_captain=ft1.captain,
        ft1_move=_move_text(set(state.ids()), set(ft_state.ids()), web),
        decision_xi_c_score_xp=decision_xp,
        hold_xi_c_score_xp=hold_xp,
        ft1_xi_c_score_xp=ft1_xp,
        step_scores=steps,
    )


def _find_bootstrap(gw: int, root: Path | None) -> Path | None:
    folder = gw_folder(gw, root)
    for relative in ("fpl_t1/bootstrap.json", "betfair_t1/bootstrap.json"):
        path = folder / relative
        if path.is_file() and not _under_live(path):
            return path
    return None


def run_numagent(
    gw: int,
    *,
    deadline_utc: str,
    root: Path | None = None,
    lock: bool = False,
    frozen_at_utc: str | None = None,
    ledger: Path | None = None,
    pull: PullFn | None = None,
    score_fn: Callable[..., NumagentPick] | None = None,
    stamp: str | None = None,
    bootstrap_path: Path | None = None,
    history_dir: Path | None = None,
) -> dict[str, Any]:
    """Compile, price, and optionally lock. A locked manifest is left as it is."""
    base = Path(root) if root is not None else ROOT
    manifest = manifest_path(gw, base)
    existing = _read_manifest(manifest)
    if existing is not None and existing.get("locked") is True:
        raise SessionError(f"locked manifest is not overwritten: {manifest}")
    ledger_path = ledger or (base / "data" / "predictions" / "2026-27" / "week_freeze.jsonl")
    if _ledger_has_gw(ledger_path, gw):
        raise SessionError(f"GW{int(gw)} already has a freeze row")
    steps: list[dict[str, Any]] = []
    sheet = gw_folder(gw, base) / "xmi_t1.csv"
    boot = bootstrap_path or _find_bootstrap(gw, base)
    if boot is None:
        raise SessionError("bootstrap for the minutes sheet must sit in the gameweek folder")
    if sheet.is_file():
        steps.append({"name": "xmi", "status": "reused", "file": sheet.name})
    else:
        sheet = compile_minutes(
            gw,
            deadline_utc=deadline_utc,
            root=base,
            bootstrap_path=boot,
            history_dir=history_dir,
            ledger=ledger_path,
            sources_path=None,
        )
        steps.append({"name": "xmi", "status": "ok", "file": sheet.name})
    folder = gw_folder(gw, base) / "betfair_t1"
    reused = book_is_ready(folder)
    exchange = resolve_exchange(gw, root=base, pull=pull)
    steps.append({"name": "exchange", "status": "reused" if reused else "pulled"})
    if score_fn is None:
        fixtures = exchange / "fixtures.json"
        odds = base / "data" / "cache" / "E0_2627.csv"
        logs = base / "data" / "cache" / "player_gw_2026_27.csv"
        entry = base / "data" / "entry" / "2632584.json"
        slate = exchange / "bootstrap.json"
        pick = select_on_score_xp(
            gw,
            minutes_path=sheet,
            exchange=exchange,
            bootstrap_path=slate if slate.is_file() else boot,
            fixtures_path=fixtures,
            entry_path=entry,
            logs_path=logs,
            odds_path=odds,
        )
    else:
        pick = score_fn(gw=gw, minutes_path=sheet, exchange=exchange)
    taken = stamp or stamp_now()
    scores_path = write_step_scores(gw, pick.step_scores, stamp=taken, root=base)
    steps.append({"name": "scores", "status": "ok", "file": scores_path.name})
    steps.append({"name": "team", "status": "ok", "chip": pick.chip})
    locked = False
    official_hash = None
    official_file = gw_folder(gw, base)
    found = sorted(official_file.glob("official_*_t1.csv"))
    if found:
        official_hash = sha256_file(found[-1])
    if lock:
        frozen = frozen_at_utc or utc_now()
        if not published_before(frozen, deadline_utc):
            raise SessionError("frozen_at is not before the deadline")
        official = official_capture(gw, base)
        lock_from_pick(
            pick,
            gw=gw,
            deadline_utc=deadline_utc,
            frozen_at_utc=frozen,
            official_path=official,
            minutes_path=sheet,
            lines_path=exchange / "gw_lines.csv",
            scores_path=scores_path,
            ledger=ledger_path,
            root=base,
        )
        locked = True
        steps.append({"name": "lock", "status": "locked"})
    else:
        steps.append({"name": "lock", "status": "skipped"})
    payload = {
        "gw": int(gw),
        "steps": steps,
        "hashes": {
            "official_sha256": official_hash,
            "minutes_sha256": sha256_file(sheet),
            "odds_manifest_sha256": sha256_file(exchange / "gw_lines.csv"),
            "pool_forecast_sha256": sha256_file(scores_path),
        },
        "locked": locked,
        "realised": None,
        "chip": pick.chip,
        "score_xp_xi_c": {
            "decision": float(pick.decision_xi_c_score_xp),
            "hold": float(pick.hold_xi_c_score_xp),
            "ft1": float(pick.ft1_xi_c_score_xp),
        },
    }
    _write_manifest(manifest, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Numagent live session")
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument("--deadline", type=str, required=True)
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--frozen-at", type=str, default="")
    parser.add_argument("--ledger", type=str, default="")
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else None
    ledger = Path(args.ledger) if args.ledger else None
    try:
        payload = run_numagent(
            int(args.gw),
            deadline_utc=args.deadline,
            root=root,
            lock=bool(args.lock),
            frozen_at_utc=args.frozen_at or None,
            ledger=ledger,
        )
    except (SessionError, ProvenanceViolationError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    json.dump({"gw": payload["gw"], "locked": payload["locked"], "chip": payload["chip"]}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
