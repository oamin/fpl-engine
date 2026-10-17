"""Research dry-runs. They do not overwrite the 22:09 capture or the minutes file.

The score stays ``score_xp``. The numeric engine is the live scorer plus the
squad MILP. A differential player has bootstrap ownership strictly under 15%.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.eval.predictions import write_prediction
from src.live.deadline import (
    BOOTSTRAP_PATH,
    ENTRY_PATH,
    FIXTURES_PATH,
    LOG_PATH,
    ODDS_PATH,
    api_selling,
    current_costs,
    final_players,
    gameweek_values,
    holdings_state,
    line_status,
    load_odds_frame,
    purchase_source,
    readiness,
    reconstruct_purchases,
    resolve_holdings,
    team_names,
)
from src.live.fpl_snapshot import ELEMENT
from src.live.news_packets import compile_player_xmi, load_gameweek_packets
from src.live.news_tags import prior_minutes
from src.live.plan import live_xi
from src.live.scorer import (
    ELIGIBLE_XMI,
    SCORE_COL,
    build_pool,
    deadline_shares,
    player_key,
    price_half,
    roster_from_bootstrap,
)
from src.models.season_climb_budget import pick_squad
from src.models.xp_engine import MIN_HISTORY
from src.rules.fpl_2026 import BUDGET_TENTHS, SQUAD_QUOTA

ROOT = Path(__file__).resolve().parents[2]
DEADLINE = "2026-10-10T10:00:00Z"
GW = 6
OWN_LIMIT = 15.0


def parse_ownership(raw: object) -> float | None:
    """Bootstrap ownership as a percent. Unparseable values are missing."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        number = float(raw)
        if number != number:
            return None
        return number
    text = str(raw).strip().rstrip("%")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_differential(percent: float | None, limit: float = OWN_LIMIT) -> bool:
    return percent is not None and percent < float(limit)


def ownership_map(bootstrap: Mapping[str, Any]) -> dict[str, float | None]:
    """Element id string → ownership percent. Missing stays missing."""
    out: dict[str, float | None] = {}
    for element in bootstrap["elements"]:
        out[str(int(element["id"]))] = parse_ownership(element.get("selected_by_percent"))
    return out


def _history(logs: pd.DataFrame, gw: int) -> dict[int, list[float]]:
    frame = logs.copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["gw"] = pd.to_numeric(frame["gw"], errors="coerce")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["player_id", "gw"])
    frame = frame.loc[frame["gw"] < int(gw)]
    grouped = frame.groupby(["player_id", "gw"], as_index=False)["minutes"].sum()
    grouped = grouped.sort_values(["player_id", "gw"])
    hist: dict[int, list[float]] = {}
    for row in grouped.itertuples(index=False):
        hist.setdefault(int(row.player_id), []).append(float(row.minutes))
    return hist


def compile_news_table(
    bootstrap: Mapping[str, Any],
    logs: pd.DataFrame,
    *,
    deadline_utc: str = DEADLINE,
    gw: int = GW,
) -> pd.DataFrame:
    """One row per element. ``xmi`` is the packet compile, not the stored file."""
    packets = load_gameweek_packets(
        gw,
        deadline_utc=deadline_utc,
        bootstrap=bootstrap,
        include_synthetic_fpl=True,
    )
    by_player: dict[int, list] = {}
    for packet in packets:
        for pid in packet.player_ids:
            by_player.setdefault(int(pid), []).append(packet)
    hist = _history(logs, gw)
    rows: list[dict[str, Any]] = []
    for element in bootstrap["elements"]:
        pid = int(element["id"])
        series = hist.get(pid, [])
        chance_raw = element.get("chance_of_playing_next_round")
        try:
            chance = None if chance_raw is None else float(chance_raw)
        except (TypeError, ValueError):
            chance = None
        compiled = compile_player_xmi(
            player_id=pid,
            name=str(element.get("web_name") or pid),
            position=ELEMENT[int(element["element_type"])],
            prior=prior_minutes(series),
            status=str(element.get("status") or ""),
            chance=chance,
            packets=by_player.get(pid, []),
            minutes=series,
        )
        xmi = compiled["xmi_compiled"]
        rows.append(
            {
                "player_id": player_key(pid),
                "element_id": pid,
                "name": str(element.get("web_name") or pid),
                "xmi": 0.0 if xmi is None else float(xmi),
                "tag": str(compiled.get("tag") or ""),
            }
        )
    return pd.DataFrame(rows)


def compiled_minutes(
    bootstrap: Mapping[str, Any],
    logs: pd.DataFrame,
    *,
    deadline_utc: str = DEADLINE,
    gw: int = GW,
) -> dict[str, float]:
    """Packet compile for every element. Keys are ``2026-27:<id>``."""
    table = compile_news_table(bootstrap, logs, deadline_utc=deadline_utc, gw=gw)
    return {str(row.player_id): float(row.xmi) for row in table.itertuples(index=False)}


def score_news_pool(
    minutes: Mapping[str, float],
    *,
    gw: int = GW,
    bootstrap: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One ``price_half`` on the news minutes. Owned players stay in the pool."""
    from src.live.t1_inputs import live_score_book

    boot = bootstrap if bootstrap is not None else json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    logs = pd.read_csv(LOG_PATH)
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    lines = live_score_book(int(gw), None)
    artifacts = None if lines is None else lines.parent
    odds = load_odds_frame(ODDS_PATH, lines)
    names = team_names(boot)
    ready, reasons = readiness(line_status(odds, fixtures, names, gw), True)
    if not ready:
        raise RuntimeError("deadline is not priced: " + ", ".join(reasons))
    players = final_players(entry)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    sources = {pid: purchase_source(entry, pid) for pid in purchases}
    holdings = resolve_holdings(
        players,
        purchases,
        current_costs(boot),
        api_selling(entry),
        sources,
    )
    state = holdings_state(holdings, int(entry["bank"]), int(entry["ft_for_next"]))
    scored = price_half(
        gw=int(gw),
        logs=logs,
        odds=odds,
        fixtures=fixtures,
        bootstrap=boot,
        state=state,
        minutes=minutes,
        played={
            int(row["gw"]): str(row["chip"])
            for row in entry.get("chips_played") or []
            if row.get("chip")
        },
        artifacts_dir=artifacts,
    )
    scores = scored.step_scores.get(int(gw)) or {}
    shares = deadline_shares(logs, int(gw))
    roster = roster_from_bootstrap(boot)
    pool = build_pool(roster, shares, minutes, set(state.ids()))
    return pool, {str(pid): float(score) for pid, score in scores.items()}


def attach_ownership(pool: pd.DataFrame, bootstrap: Mapping[str, Any]) -> pd.DataFrame:
    own = {
        player_key(int(element["id"])): parse_ownership(element.get("selected_by_percent"))
        for element in bootstrap["elements"]
    }
    names = {
        player_key(int(element["id"])): str(element["web_name"])
        for element in bootstrap["elements"]
    }
    frame = pool.copy()
    frame["ownership"] = frame["player_id"].map(own)
    frame["name"] = frame["player_id"].map(names)
    frame[SCORE_COL] = pd.to_numeric(frame[SCORE_COL], errors="coerce")
    return frame


def apply_scores(pool: pd.DataFrame, scores: Mapping[str, float]) -> pd.DataFrame:
    """Write ``score_xp`` from the priced week. A missing score stays missing."""
    frame = pool.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame[SCORE_COL] = frame["player_id"].map(lambda pid: scores.get(str(pid)))
    frame[SCORE_COL] = pd.to_numeric(frame[SCORE_COL], errors="coerce")
    return frame


def milp_squad(frame: pd.DataFrame) -> pd.DataFrame:
    """Fresh 15. The owned-player bypass is already applied by the caller."""
    work = frame.loc[frame["eligible"].astype(bool)].copy()
    work = work.loc[pd.to_numeric(work[SCORE_COL], errors="coerce").notna()]
    if "value" not in work.columns:
        raise RuntimeError("pool has no value")
    return pick_squad(work, SCORE_COL, budget=int(BUDGET_TENTHS))


def differential_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Eligible buyers under 15%. Owned players do not skip the gates."""
    work = frame.loc[frame["eligible"].astype(bool)].copy()
    work = work.loc[
        work["ownership"].map(lambda value: is_differential(None if pd.isna(value) else float(value)))
    ]
    work = work.loc[pd.to_numeric(work["n_prior"], errors="coerce").fillna(0) >= MIN_HISTORY]
    work = work.loc[pd.to_numeric(work["minutes"], errors="coerce").fillna(0) >= ELIGIBLE_XMI]
    return work


def position_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "position" not in frame.columns:
        return {pos: 0 for pos in SQUAD_QUOTA}
    counts = frame["position"].astype(str).value_counts().to_dict()
    return {pos: int(counts.get(pos, 0)) for pos in SQUAD_QUOTA}


def cheapest_quota_cost(frame: pd.DataFrame) -> int | None:
    """Cheapest exact quota, ignoring the club cap. None if a position is short."""
    if frame.empty:
        return None
    total = 0
    for pos, need in SQUAD_QUOTA.items():
        sub = frame.loc[frame["position"].astype(str) == pos]
        prices = pd.to_numeric(sub["value"], errors="coerce").dropna().sort_values()
        if len(prices) < int(need):
            return None
        total += int(prices.iloc[: int(need)].sum())
    return total


def write_score_csv(scores: Mapping[str, float], path: Path, created_at: str) -> None:
    rows = [
        {
            "player_id": pid,
            "gw": GW,
            "score": float(score),
            "created_at": created_at,
            "note": "news xmi dry run",
        }
        for pid, score in sorted(scores.items())
    ]
    write_prediction(path, pd.DataFrame(rows))


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, AttributeError):
            pass
    return value


def run_numeric(
    *,
    created_at: str,
    score_path: Path,
    old_minutes_path: Path | None = None,
) -> dict[str, Any]:
    """News-informed ``score_xp``, then a fresh fifteen and a differential fifteen."""
    boot = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    logs = pd.read_csv(LOG_PATH)
    table = compile_news_table(boot, logs)
    minutes = {str(row.player_id): float(row.xmi) for row in table.itertuples(index=False)}
    pool, scores = score_news_pool(minutes, bootstrap=boot)
    framed = apply_scores(attach_ownership(pool, boot), scores)
    tags = {str(row.player_id): str(row.tag) for row in table.itertuples(index=False)}
    framed["tag"] = framed["player_id"].astype(str).map(tags)
    fresh = milp_squad(framed)
    fresh_lines = squad_lines(fresh)
    diff = differential_frame(framed)
    differential: dict[str, Any]
    try:
        picked = milp_squad(diff)
        differential = {"ok": True, **squad_lines(picked)}
    except RuntimeError as exc:
        differential = {
            "ok": False,
            "error": str(exc),
            "n": int(len(diff)),
            "by_position": position_counts(diff),
            "cheapest_quota": cheapest_quota_cost(diff),
        }
    if score_path.exists():
        raise FileExistsError(f"refusing to overwrite {score_path}")
    write_score_csv(scores, score_path, created_at)
    old_xmi: dict[int, float] = {}
    if old_minutes_path is not None and old_minutes_path.is_file():
        stored = pd.read_csv(old_minutes_path)
        for row in stored.itertuples(index=False):
            old_xmi[int(row.player_id)] = float(row.xmi)
    watch = table.copy()
    watch["old_xmi"] = watch["element_id"].map(old_xmi)
    moved = watch.loc[
        watch["old_xmi"].notna() & ((watch["xmi"] - watch["old_xmi"]).abs() > 0.05)
    ]
    return {
        "created_at": created_at,
        "score_path": str(score_path),
        "n_scored": len(scores),
        "n_pool": int(len(framed)),
        "n_eligible": int(framed["eligible"].astype(bool).sum()),
        "n_differential": int(len(diff)),
        "differential_by_position": position_counts(diff),
        "fresh": fresh_lines,
        "differential": differential,
        "minutes_moved": int(len(moved)),
        "watch": [
            {
                "element_id": int(row.element_id),
                "name": str(row.name),
                "xmi": float(row.xmi),
                "tag": str(row.tag),
                "old_xmi": None if pd.isna(row.old_xmi) else float(row.old_xmi),
            }
            for row in watch.itertuples(index=False)
            if str(row.name)
            in {
                "Haaland",
                "Wood",
                "Shaw",
                "Van Ewijk",
                "Scarlett",
                "Ødegaard",
                "Calvert-Lewin",
                "Salah",
                "Palmer",
                "Saka",
            }
        ],
    }


def squad_lines(squad: pd.DataFrame, score_col: str = SCORE_COL) -> dict[str, Any]:
    picked = live_xi(squad, score_col)
    xi = picked["xi"]
    captain_row = xi.loc[xi["player_id"].astype(str) == picked["captain"]].iloc[0]
    captain_score = float(captain_row[score_col])
    return {
        "formation": picked["formation"],
        "captain": str(picked["captain"]),
        "vice": str(picked["vice"]),
        "xi_score": float(picked["xi_score"]),
        "xi_captain": float(picked["xi_score"]) + captain_score,
        "players": squad.sort_values(["position", "name"]).to_dict(orient="records"),
        "xi_ids": [str(pid) for pid in xi["player_id"].tolist()],
        "bench_ids": [str(pid) for pid in picked["bench"]["player_id"].tolist()],
    }


def main() -> None:
    from datetime import datetime, timezone

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    score_path = ROOT / "data" / "predictions" / "2026-27" / "gw06" / f"news_xmi_{stamp}.csv"
    result = run_numeric(
        created_at=created,
        score_path=score_path,
        old_minutes_path=ROOT / "data" / "live" / "xmi_gw6.csv",
    )
    dest = ROOT / "reports" / "news_xmi_dry_run_20261008.json"
    dest.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(dest)
    print("eligible", result["n_eligible"], "differential", result["n_differential"])
    print("fresh", result["fresh"]["formation"], result["fresh"]["xi_captain"])
    print("differential_ok", result["differential"].get("ok"), result["differential"].get("error"))


if __name__ == "__main__":
    main()
