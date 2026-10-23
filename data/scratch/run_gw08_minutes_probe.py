"""GW8 full minutes_llm probe: cases + FFScout lineups + tags → xmi."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from src.live.ffscout_team_news import match_ffs_name_to_fpl, pull_team_news
from src.live.minutes_llm import apply_completion, build_prompt, required_cases
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "data/scratch"
GW = 8
DEADLINE = "2026-10-23T17:30:00Z"
ENTRY = 2632584


def main() -> int:
    boot = httpx.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=60
    ).json()
    live = httpx.get(
        f"https://fantasy.premierleague.com/api/event/{GW - 1}/live/", timeout=60
    ).json()
    picks = httpx.get(
        f"https://fantasy.premierleague.com/api/entry/{ENTRY}/event/{GW - 1}/picks/",
        timeout=60,
    ).json()
    owned = {int(p["element"]) for p in picks["picks"]}

    teams = {int(t["id"]): t["name"] for t in boot["teams"]}
    els = {int(e["id"]): e for e in boot["elements"]}
    elements = boot["elements"]

    hist = pd.read_csv(ROOT / "data/cache/player_gw_2026_27.csv")
    hist = hist.loc[pd.to_numeric(hist["gw"], errors="coerce") < GW - 1].copy()
    live_rows = []
    for block in live.get("elements") or []:
        eid = int(block["id"])
        el = els.get(eid)
        if el is None:
            continue
        st = block.get("stats") or {}
        live_rows.append(
            {
                "gw": GW - 1,
                "player_id": eid,
                "minutes": float(st.get("minutes") or 0),
            }
        )
    hist = pd.concat([hist, pd.DataFrame(live_rows)], ignore_index=True)

    last_minutes: dict[int, float] = {}
    minutes_history: dict[int, list[float]] = {}
    for eid, g in hist.groupby("player_id"):
        g = g.sort_values("gw")
        mins = pd.to_numeric(g["minutes"], errors="coerce").fillna(0.0).tolist()
        minutes_history[int(eid)] = mins
        if mins:
            last_minutes[int(eid)] = float(mins[-1])

    cases = required_cases(
        boot,
        owned=owned,
        last_minutes=last_minutes,
        minutes_history=minutes_history,
        expand_club_squads=True,
    )
    print(
        f"cases={len(cases)} focus={sum(1 for c in cases if c.focus)} "
        f"clubs={len({c.club for c in cases})}"
    )

    prompt = build_prompt(cases, GW)
    (SCRATCH / "gw08_minutes_prompt.txt").write_text(prompt, encoding="utf-8")

    ff = pull_team_news(out_dir=SCRATCH / "ffscout_gw08")
    print("ffscout clubs", ff["n_clubs"], "at", ff["fetched_at"])

    fpl_clubs = {c.club for c in cases}
    club_alias = {norm_team(c): c for c in fpl_clubs}
    for short, full_hint in [
        ("Coventry", "Coventry City"),
        ("Hull", "Hull City"),
        ("Ipswich", "Ipswich Town"),
    ]:
        if full_hint in fpl_clubs:
            club_alias[norm_team(short)] = full_hint

    by_club_els: dict[str, list[dict]] = {}
    for el in elements:
        by_club_els.setdefault(teams[int(el["team"])], []).append(el)

    def resolve_name(name: str, club_fpl: str) -> int | None:
        cands = by_club_els.get(club_fpl, [])
        hit = match_ffs_name_to_fpl(name, cands)
        if hit is not None:
            return int(hit["id"])
        # Bare surname ties (e.g. Newcastle Miley → L.Miley vs Mason): prefer
        # the player with more recent minutes.
        from src.live.ffscout_team_news import _norm_player

        key = _norm_player(name)
        if " " in key:
            return None
        tied = []
        for el in cands:
            second = _norm_player(el.get("second_name") or "")
            web = _norm_player(el.get("web_name") or "")
            second_last = second.split()[-1] if second else ""
            if key and (key == second or key == second_last or web.endswith(key)):
                tied.append(el)
        if not tied:
            return None
        tied.sort(
            key=lambda el: (
                -float(last_minutes.get(int(el["id"]), 0.0)),
                int(el["id"]),
            )
        )
        return int(tied[0]["id"])

    players_out = []
    for case in cases:
        tag = "none"
        note = ""
        if case.status in ("s", "u") or (case.chance is not None and case.chance == 0):
            tag = "injured"
            note = case.news[:80] or f"status={case.status}"
        elif case.chance is not None and case.chance <= 25:
            tag = "injured"
            note = f"chance {case.chance:g}; {case.news[:60]}"
        elif case.chance is not None and case.chance <= 50:
            tag = "50/50"
            note = f"chance {case.chance:g}; {case.news[:60]}"
        elif case.news and any(
            w in case.news.lower() for w in ("expected back", "injury", "knock", "doubt")
        ):
            if case.chance is not None and case.chance <= 75:
                tag = "50/50"
                note = case.news[:80]
            elif "expected back" in case.news.lower() and "nov" in case.news.lower():
                tag = "injured"
                note = case.news[:80]
        players_out.append(
            {"player_id": case.player_id, "tag": tag, "note": note}
        )

    by_id_tag = {p["player_id"]: p for p in players_out}
    lineups_out: list[dict] = []
    for club_row in ff["clubs"]:
        fpl_club = club_alias.get(norm_team(club_row["club"]))
        if fpl_club is None:
            if club_row["club"] in fpl_clubs:
                fpl_club = club_row["club"]
            else:
                print("skip ffscout club", club_row["club"])
                continue
        for name in club_row["out"] + club_row["banned"]:
            pid = resolve_name(name, fpl_club)
            if pid is not None and pid in by_id_tag:
                by_id_tag[pid]["tag"] = "injured"
                by_id_tag[pid]["note"] = (
                    by_id_tag[pid]["note"] + f"; ffscout out/ban:{name}"
                ).strip("; ")
        for d in club_row["doubts"]:
            name = d["name"]
            pid = resolve_name(name, fpl_club)
            if pid is None or pid not in by_id_tag:
                continue
            if by_id_tag[pid]["tag"] == "injured":
                continue
            by_id_tag[pid]["tag"] = "50/50"
            by_id_tag[pid]["note"] = (
                by_id_tag[pid]["note"]
                + f"; ffscout doubt:{name} {d.get('chance') or ''}"
            ).strip("; ")

        xi_ids: list[int] = []
        for name in club_row["predicted_xi"]:
            pid = resolve_name(name, fpl_club)
            if pid is None:
                print(f"  unmatched XI {fpl_club}: {name}")
                xi_ids = []
                break
            xi_ids.append(pid)
        if len(xi_ids) == 11 and len(set(xi_ids)) == 11:
            lineups_out.append({"club": fpl_club, "xi": xi_ids, "bench": []})
            print(f"lineup ok {fpl_club}")
        else:
            print(f"lineup skip {fpl_club} n={len(xi_ids)}")

    payload = {"players": list(by_id_tag.values()), "lineups": lineups_out}
    (SCRATCH / "gw08_minutes_llm_raw.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    cases_dump = [
        {
            "player_id": c.player_id,
            "web_name": c.web_name,
            "position": c.position,
            "club": c.club,
            "status": c.status,
            "chance": c.chance,
            "news": c.news,
            "last_minutes": c.last_minutes,
            "focus": c.focus,
        }
        for c in cases
    ]
    (SCRATCH / "gw08_minutes_cases.json").write_text(
        json.dumps(cases_dump, indent=2), encoding="utf-8"
    )
    (SCRATCH / "gw08_minutes_meta.json").write_text(
        json.dumps(
            {
                "gw": GW,
                "deadline_utc": DEADLINE,
                "n_cases": len(cases),
                "n_focus": sum(1 for c in cases if c.focus),
                "clubs": sorted({c.club for c in cases}),
                "ffscout_fetched_at": ff["fetched_at"],
                "n_lineups": len(lineups_out),
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "method": "minutes_llm + FFScout predicted XI / out / doubts",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = apply_completion(
        json.dumps(payload),
        cases,
        gw=GW,
        csv_path=SCRATCH / "gw08_xmi_probe.csv",
        report_path=SCRATCH / "gw08_minutes_probe.md",
    )
    print("ok", result.ok, "n_errors", len(result.errors))
    if not result.ok:
        for e in result.errors[:40]:
            print(" ERR", e)
        return 1
    print("rows", len(result.rows), "lineups", len(result.lineups))
    for pid in sorted(owned):
        row = next(r for r in result.rows if r["player_id"] == pid)
        el = els[pid]
        print(
            f"  {el['web_name']:16} xmi={float(row['xmi']):5.1f} "
            f"tag={row['tag']:12} src={row['source']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
