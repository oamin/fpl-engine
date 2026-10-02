"""Penalty shots from Understat, joined onto the Vaastav gameweek sheets.

The shot feed is a post-match fact: player, result, xG, and the match date.
It is not a pre-deadline designated-taker list. ``score_xp`` is left unchanged.

A shot is written onto a row only when the taker's name and club match exactly
one FPL player. A tied or unknown name is left blank. Confirmed zeroes are
written only for a fixture whose every penalty shot joined.
"""

from __future__ import annotations

import csv
import html
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

UNDERSTAT = "https://understat.com"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
PLAYERS_RAW = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/"
    "data/{season}/players_raw.csv"
)

# Understat season year, Vaastav season label, cache stem.
SEASONS: tuple[tuple[str, str, str], ...] = (
    ("2022", "2022-23", "2022_23"),
    ("2023", "2023-24", "2023_24"),
    ("2024", "2024-25", "2024_25"),
    ("2025", "2025-26", "2025_26"),
)

SHEET_COLUMNS = ("penalties_taken", "penalties_scored", "penalty_xg")
# Understat prints the legal name. FPL's squad list uses the shorter one.
# The replacement must itself join to exactly one player. It is not a fuzzy rule.
SPELLINGS = {
    "eli junior kroupi": "junior kroupi",
}
PROGRESS = CACHE / "understat_match_pens.jsonl"

_SPECIAL = str.maketrans(
    {
        "ø": "o",
        "Ø": "o",
        "æ": "ae",
        "Æ": "ae",
        "ł": "l",
        "Ł": "l",
        "đ": "d",
        "Đ": "d",
        "ß": "ss",
        "ð": "d",
        "ı": "i",
    }
)


@dataclass(frozen=True)
class RosterPlayer:
    element: int
    name: str
    team: str
    web_name: str
    first_name: str
    second_name: str
    date: str = ""


def fold_person(name: str) -> str:
    """Case-fold a person name. Ø and accents collapse. Apostrophes drop."""
    text = html.unescape(str(name)).translate(_SPECIAL)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace("’", "")
    text = text.replace("-", " ").replace(".", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _initial_web(understat_tokens: list[str], web_name: str) -> bool:
    """'Bruno Fernandes' matches a web name 'B.Fernandes'."""
    web_tokens = fold_person(web_name).split()
    if len(understat_tokens) < 2 or len(web_tokens) < 2:
        return False
    if web_tokens[-1] != understat_tokens[-1]:
        return False
    initials = [tok for tok in web_tokens[:-1] if len(tok) == 1]
    given = understat_tokens[:-1]
    if len(initials) != len(given):
        return False
    return all(given[i].startswith(initials[i]) for i in range(len(initials)))


def _suffix_web(understat_tokens: list[str], web_name: str, first_name: str) -> bool:
    """'Dominic Solanke' matches web name 'Solanke' when the given name agrees."""
    web_tokens = fold_person(web_name).split()
    if not understat_tokens or not web_tokens or len(web_tokens) > len(understat_tokens):
        return False
    if understat_tokens[-len(web_tokens) :] != web_tokens:
        return False
    rest = understat_tokens[: -len(web_tokens)]
    if not rest:
        return True
    first_tokens = fold_person(first_name).split()
    if not first_tokens:
        return False
    return rest[0] == first_tokens[0] or (
        len(rest[0]) >= 3 and first_tokens[0].startswith(rest[0])
    )


def _matches(understat_name: str, player: RosterPlayer) -> bool:
    folded = fold_person(understat_name)
    tokens = folded.split()
    full = fold_person(player.name)
    web = fold_person(player.web_name)
    legal = fold_person(f"{player.first_name} {player.second_name}")
    if folded and folded in {full, web, legal}:
        return True
    if tokens and all(tok in full.split() for tok in tokens):
        return True
    if _initial_web(tokens, player.web_name):
        return True
    if _suffix_web(tokens, player.web_name, player.first_name):
        return True
    return False


def match_element(understat_name: str, team: str, roster: list[RosterPlayer]) -> int | None:
    """Unique FPL element for this taker and club. Ties and misses return None."""
    club = norm_team(team)
    names = [understat_name]
    alias = SPELLINGS.get(fold_person(understat_name))
    if alias:
        names.append(alias)
    hits: list[int] = []
    for name in names:
        hits.extend(p.element for p in roster if p.team == club and _matches(name, p))
    unique = sorted(set(hits))
    if len(unique) != 1:
        return None
    return unique[0]


def _ajax_headers(referer: str) -> dict[str, str]:
    return {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": referer,
    }


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA},
        follow_redirects=True,
        timeout=40.0,
    )


def fetch_results(season_year: str, client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Played matches for one Understat season. Needs the league-page cookie."""
    own = client is None
    client = client or _client()
    try:
        page = f"{UNDERSTAT}/league/EPL/{season_year}"
        client.get(page)
        payload = client.get(
            f"{UNDERSTAT}/getLeagueData/EPL/{season_year}",
            headers=_ajax_headers(page),
        )
        payload.raise_for_status()
        dates = payload.json()["dates"]
        return [row for row in dates if row.get("isResult")]
    finally:
        if own:
            client.close()


def penalty_rows_from_match(payload: dict[str, Any], season_label: str) -> list[dict[str, Any]]:
    """Penalty shots only. Other situations are dropped."""
    rows: list[dict[str, Any]] = []
    shots = payload.get("shots") or {}
    for side in ("h", "a"):
        for shot in shots.get(side) or []:
            if shot.get("situation") != "Penalty":
                continue
            team = shot.get("h_team") if shot.get("h_a") == "h" else shot.get("a_team")
            opponent = shot.get("a_team") if shot.get("h_a") == "h" else shot.get("h_team")
            rows.append(
                {
                    "season": season_label,
                    "match_id": str(shot.get("match_id") or ""),
                    "date": str(shot.get("date") or "")[:10],
                    "team": str(team or ""),
                    "opponent": str(opponent or ""),
                    "player": str(shot.get("player") or ""),
                    "player_id": str(shot.get("player_id") or ""),
                    "minute": shot.get("minute"),
                    "result": str(shot.get("result") or ""),
                    "xg": float(shot.get("xG") or 0.0),
                    "shot_id": str(shot.get("id") or ""),
                }
            )
    return rows


def _fetch_chunk(chunk: list[tuple[str, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """One cookie session for a batch of matches."""
    rows: list[dict[str, Any]] = []
    warmed: set[str] = set()
    with _client() as client:
        for season_year, season_label, match in chunk:
            match_id = str(match["id"])
            home = match["h"]["title"]
            away = match["a"]["title"]
            date = str(match.get("datetime") or "")[:10]
            if season_year not in warmed:
                client.get(f"{UNDERSTAT}/league/EPL/{season_year}")
                warmed.add(season_year)
            last_error = "no attempt"
            ok = False
            pens: list[dict[str, Any]] = []
            for _ in range(3):
                try:
                    response = client.get(
                        f"{UNDERSTAT}/getMatchData/{match_id}",
                        headers=_ajax_headers(f"{UNDERSTAT}/match/{match_id}"),
                    )
                    response.raise_for_status()
                    pens = penalty_rows_from_match(response.json(), season_label)
                    ok = True
                    break
                except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
            row: dict[str, Any] = {
                "ok": ok,
                "season": season_label,
                "season_year": season_year,
                "match_id": match_id,
                "date": date,
                "home": home,
                "away": away,
                "pens": pens,
            }
            if not ok:
                row["error"] = last_error
            rows.append(row)
    return rows


def load_progress() -> dict[str, dict[str, Any]]:
    if not PROGRESS.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in PROGRESS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["match_id"])] = row
    return out


def _write_progress(done: dict[str, dict[str, Any]]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(done[key]) for key in sorted(done)]
    PROGRESS.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def fetch_all(workers: int = 6) -> dict[str, dict[str, Any]]:
    """Download every played EPL match's penalty shots. Successful caches are skipped."""
    done = {key: row for key, row in load_progress().items() if row.get("ok")}
    todo: list[tuple[str, str, dict[str, Any]]] = []
    for year, label, _stem in SEASONS:
        with _client() as client:
            matches = fetch_results(year, client)
        for match in matches:
            if str(match["id"]) not in done:
                todo.append((year, label, match))
    if not todo:
        return done
    print(f"fetching {len(todo)} matches", flush=True)
    chunks: list[list[tuple[str, str, dict[str, Any]]]] = [[] for _ in range(workers)]
    for index, item in enumerate(todo):
        chunks[index % workers].append(item)
    finished = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_fetch_chunk, chunk) for chunk in chunks if chunk]
        for fut in as_completed(futures):
            for row in fut.result():
                done[str(row["match_id"])] = row
                finished += 1
            print(f"  {finished}/{len(todo)}", flush=True)
            _write_progress(done)
    return done


def _cache_players_raw(season_label: str, stem: str) -> Path:
    path = CACHE / f"players_raw_{stem}.csv"
    if path.exists():
        return path
    text = httpx.get(PLAYERS_RAW.format(season=season_label), timeout=60.0, follow_redirects=True).text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_roster(stem: str, season_label: str) -> list[RosterPlayer]:
    raw_path = _cache_players_raw(season_label, stem)
    raw = list(csv.DictReader(raw_path.read_text(encoding="utf-8").splitlines()))
    by_id = {row["id"]: row for row in raw}
    sheet = CACHE / f"merged_gw_{stem}.csv"
    roster: list[RosterPlayer] = []
    with sheet.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            element = int(row["element"])
            extra = by_id.get(str(element), {})
            roster.append(
                RosterPlayer(
                    element=element,
                    name=row.get("name") or "",
                    team=norm_team(row.get("team") or ""),
                    web_name=extra.get("web_name") or "",
                    first_name=extra.get("first_name") or "",
                    second_name=extra.get("second_name") or "",
                    date=_kickoff_date(row.get("kickoff_time") or ""),
                )
            )
    return roster


def _pool(roster: list[RosterPlayer], date: str, team: str) -> list[RosterPlayer]:
    """Players at that club on that kickoff. Undated rows are the unit-test roster."""
    club = norm_team(team)
    return [p for p in roster if p.team == club and (p.date == "" or p.date == date)]


def _format_xg(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if value else "0"


def _cell(taken: int, scored: int, xg: float) -> tuple[str, str, str]:
    return (str(taken), str(scored), _format_xg(xg))


def fixture_cells(
    matches: list[dict[str, Any]],
    roster: list[RosterPlayer],
    sheet_rows: list[tuple[str, int, str]],
) -> tuple[dict[tuple[str, int], tuple[str, str, str]], list[dict[str, Any]]]:
    """Map (date, element) to sheet cells. Unresolved fixtures do not get zeroes.

    ``sheet_rows`` is (kickoff date, element, normalised team).
    """
    by_fixture: dict[tuple[str, str], list[int]] = {}
    for date, element, team in sheet_rows:
        by_fixture.setdefault((date, team), []).append(element)

    cells: dict[tuple[str, int], tuple[str, str, str]] = {}
    unmatched: list[dict[str, Any]] = []
    for match in matches:
        if not match.get("ok", True):
            continue
        date = str(match["date"])
        home = norm_team(match["home"])
        away = norm_team(match["away"])
        pens = match.get("pens") or []
        takers: dict[int, list[float]] = {}
        unresolved = False
        for shot in pens:
            element = match_element(shot["player"], shot["team"], _pool(roster, date, shot["team"]))
            if element is None:
                unresolved = True
                unmatched.append({**shot, "home": match["home"], "away": match["away"]})
                continue
            bucket = takers.setdefault(element, [0, 0, 0.0])
            bucket[0] += 1
            bucket[1] += 1 if shot["result"] == "Goal" else 0
            bucket[2] += float(shot["xg"])
        involved = []
        for club in (home, away):
            involved.extend(by_fixture.get((date, club), []))
        if not involved:
            continue
        if unresolved:
            for element, (taken, scored, xg) in takers.items():
                cells[(date, element)] = _cell(int(taken), int(scored), float(xg))
            continue
        seen: set[int] = set()
        for element in involved:
            if element in seen:
                continue
            seen.add(element)
            if element in takers:
                taken, scored, xg = takers[element]
                cells[(date, element)] = _cell(int(taken), int(scored), float(xg))
            else:
                cells[(date, element)] = ("0", "0", "0")
    return cells, unmatched


def _kickoff_date(value: str) -> str:
    return (value or "")[:10]


def _miss_agreement(stem: str) -> tuple[int, int]:
    """Non-goals on joined taker rows versus Vaastav ``penalties_missed``."""
    path = CACHE / f"merged_gw_{stem}.csv"
    ours = fpl = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            taken_text = row.get("penalties_taken") or ""
            if taken_text in {"", "0"}:
                continue
            taken = int(taken_text)
            scored = int(row["penalties_scored"])
            ours += taken - scored
            fpl += int(float(row.get("penalties_missed") or 0))
    return ours, fpl


def append_penalty_columns(
    raw_lines: list[str],
    cells: dict[tuple[str, int], tuple[str, str, str]],
) -> tuple[list[str], dict[str, int]]:
    """Append the three columns onto raw CSV lines. Earlier cells stay byte-for-byte."""
    if not raw_lines:
        return raw_lines, {"filled": 0, "blank": 0, "rows": 0}
    header = raw_lines[0]
    suffix = "," + ",".join(SHEET_COLUMNS)
    if header.endswith(suffix):
        header = header[: -len(suffix)]
        body = [line.rsplit(",", 3)[0] for line in raw_lines[1:]]
    else:
        body = raw_lines[1:]
    filled = blank = 0
    out_lines = [header + suffix]
    reader = csv.DictReader([header, *body])
    for line, row in zip(body, reader):
        key = (_kickoff_date(row.get("kickoff_time") or ""), int(row["element"]))
        values = cells.get(key)
        if values is None:
            out_lines.append(line + ",,,")
            blank += 1
        else:
            out_lines.append(line + "," + ",".join(values))
            filled += 1
    return out_lines, {"filled": filled, "blank": blank, "rows": len(body)}


def apply_sheet(stem: str, cells: dict[tuple[str, int], tuple[str, str, str]]) -> dict[str, int]:
    """Append the three penalty columns. Existing sheet cells are not rewritten."""
    path = CACHE / f"merged_gw_{stem}.csv"
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    out_lines, stats = append_penalty_columns(raw_lines, cells)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return stats


def _sheet_rows(stem: str) -> list[tuple[str, int, str]]:
    path = CACHE / f"merged_gw_{stem}.csv"
    rows: list[tuple[str, int, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    _kickoff_date(row.get("kickoff_time") or ""),
                    int(row["element"]),
                    norm_team(row.get("team") or ""),
                )
            )
    return rows


def write_fact_table(matches: dict[str, dict[str, Any]], assigned: dict[str, list[dict[str, Any]]]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "understat_penalty_shots.csv"
    fields = [
        "season",
        "match_id",
        "date",
        "team",
        "opponent",
        "player",
        "player_id",
        "minute",
        "result",
        "xg",
        "shot_id",
        "element",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for shots in assigned.values():
            for shot in shots:
                writer.writerow({key: shot.get(key, "") for key in fields})
    # Keep the match index so a fixture with no penalty is still a fetched fixture.
    index = PROCESSED / "understat_matches.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["season", "match_id", "date", "home", "away", "ok", "n_pens", "error"],
        )
        writer.writeheader()
        for row in sorted(matches.values(), key=lambda item: (item["season"], item["date"], item["match_id"])):
            writer.writerow(
                {
                    "season": row["season"],
                    "match_id": row["match_id"],
                    "date": row["date"],
                    "home": row["home"],
                    "away": row["away"],
                    "ok": row.get("ok", True),
                    "n_pens": len(row.get("pens") or []),
                    "error": row.get("error", ""),
                }
            )


def fixture_hit_rate(matches: list[dict[str, Any]], sheet_rows: list[tuple[str, int, str]]) -> float:
    """Share of fetched matches whose date and club appear on the sheet."""
    keys = {(date, team) for date, _element, team in sheet_rows}
    played = [row for row in matches if row.get("ok", True)]
    if not played:
        return 0.0
    hits = 0
    for row in played:
        home = norm_team(row["home"])
        away = norm_team(row["away"])
        if (row["date"], home) in keys or (row["date"], away) in keys:
            hits += 1
    return hits / len(played)


def run(workers: int = 6) -> str:
    matches = fetch_all(workers=workers)
    for _year, label, stem in SEASONS:
        season_matches = [row for row in matches.values() if row["season"] == label]
        rate = fixture_hit_rate(season_matches, _sheet_rows(stem))
        if season_matches and rate < 0.9:
            raise RuntimeError(
                f"{label} date join hit {rate:.1%} of Understat matches. "
                "Sheets were not changed."
            )
    lines = ["# Penalty columns on the Vaastav sheets", ""]
    lines.append(
        "Source: Understat `getMatchData` shots with `situation == Penalty`. "
        "Joined on the kickoff date and the club the player was at that day. "
        "A shot is written only when that club's squad has exactly one matching name. "
        "The one registered spelling is Eli Junior Kroupi to Junior Kroupi. "
        "A blank cell would mean that fixture was not fully joined. "
        "These are post-match facts, not a pre-deadline taker list. "
        "`score_xp` was not changed."
    )
    lines.append("")
    all_assigned: dict[str, list[dict[str, Any]]] = {}
    all_unmatched: list[dict[str, Any]] = []
    for year, label, stem in SEASONS:
        roster = load_roster(stem, label)
        season_matches = [row for row in matches.values() if row["season"] == label]
        cells, unmatched = fixture_cells(season_matches, roster, _sheet_rows(stem))
        stats = apply_sheet(stem, cells)
        assigned_shots: list[dict[str, Any]] = []
        for match in season_matches:
            for shot in match.get("pens") or []:
                pool = _pool(roster, str(match["date"]), shot["team"])
                element = match_element(shot["player"], shot["team"], pool)
                assigned_shots.append({**shot, "element": "" if element is None else element})
        all_assigned[label] = assigned_shots
        all_unmatched.extend(unmatched)
        taken = sum(int(v[0]) for v in cells.values())
        scored = sum(int(v[1]) for v in cells.values())
        failed = sum(1 for row in season_matches if not row.get("ok", True))
        our_miss, fpl_miss = _miss_agreement(stem)
        lines.append(
            f"- {label}: matches {len(season_matches)} (failed {failed}), "
            f"penalty shots {sum(len(row.get('pens') or []) for row in season_matches)}, "
            f"unmatched shots {len(unmatched)}, "
            f"sheet rows filled {stats['filled']} / blank {stats['blank']}, "
            f"penalties taken {taken}, scored {scored}. "
            f"Misses on joined takers: sheet {our_miss}, Vaastav penalties_missed {fpl_miss}."
        )
    write_fact_table(matches, all_assigned)
    unmatched_path = PROCESSED / "understat_penalty_unmatched.csv"
    fields = ["season", "date", "team", "player", "result", "xg", "match_id", "home", "away"]
    with unmatched_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for shot in all_unmatched:
            writer.writerow({key: shot.get(key, "") for key in fields})
    if all_unmatched:
        lines.append("")
        lines.append("Unmatched shots (left blank, not guessed):")
        for shot in all_unmatched:
            lines.append(
                f"- {shot['season']} {shot['date']} {shot['team']} {shot['player']} "
                f"{shot['result']} xG {shot['xg']}"
            )
    lines.append("")
    text = "\n".join(lines) + "\n"
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "penalty_sheets.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    print(run(), flush=True)


if __name__ == "__main__":
    main()
